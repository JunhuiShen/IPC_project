#include "parallel_helper.h"
#include "quaternion_math.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

bool periodic_angle_in_interval(
    double angle, double interval_min, double interval_max) {
    constexpr double two_pi = 2.0 * M_PI;
    const double shifted =
        angle + two_pi * std::ceil((interval_min - angle) / two_pi);
    const double scale = std::max(
        {1.0, std::abs(interval_min), std::abs(interval_max),
         std::abs(shifted)});
    const double tolerance =
        32.0 * std::numeric_limits<double>::epsilon() * scale;
    return shifted <= interval_max + tolerance;
}

} // namespace

AABB arc_node_aabb(
    const Vec3& x_com, const Vec4& q,
    const Vec3& X, const Vec4& q_rel) {
    if (!x_com.allFinite() || !X.allFinite()) {
        throw std::invalid_argument(
            "arc_node_aabb requires finite positions");
    }

    const Vec4 q_current = quaternion_normalize(q);
    const Vec4 q_relative = quaternion_normalize(q_rel);

    const Vec3 world_space_offset = quaternion_rotate(q_current, X);
    const Vec3 x = x_com + world_space_offset;
    const Vec3 vector_part = q_relative.tail<3>();
    const double sin_half_angle = vector_part.norm();
    if (sin_half_angle < 1.0e-12) {
        if (q_relative[0] >= 0.0)
            return AABB(x, x);
        // An exact 2*pi endpoint quaternion has lost its rotation axis. The
        // sphere box is conservative for every possible full-turn axis.
        const Vec3 radius = Vec3::Constant(X.norm());
        return AABB(x_com - radius, x_com + radius);
    }

    const Vec3 axis = vector_part / sin_half_angle;
    const double angular_extent = 2.0 * std::atan2(sin_half_angle, q_relative[0]);

    // Rodrigues' formula gives
    // p(t) = circle_center + cosine_coefficient cos(t)
    //                       + sine_coefficient sin(t).
    const Vec3 axial_offset = axis * axis.dot(world_space_offset);
    const Vec3 circle_center = x_com + axial_offset;
    const Vec3 cosine_coefficient = world_space_offset - axial_offset;
    const Vec3 sine_coefficient = axis.cross(world_space_offset);

    const auto point_at = [&](double angle) {
        return circle_center
            + std::cos(angle) * cosine_coefficient
            + std::sin(angle) * sine_coefficient;
    };

    AABB box;
    box.expand(point_at(-angular_extent));
    box.expand(point_at(angular_extent));

    // Each coordinate is c + a cos(t) + b sin(t). Its extrema occur at
    // atan2(b, a) and that angle plus pi, modulo 2*pi.
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        const double a = cosine_coefficient[coordinate];
        const double b = sine_coefficient[coordinate];
        const double amplitude = std::hypot(a, b);
        if (amplitude < 1.0e-12)
            continue;

        const double maximum_angle = std::atan2(b, a);
        if (periodic_angle_in_interval(
                maximum_angle, -angular_extent, angular_extent)) {
            box.max[coordinate] = std::max(
                box.max[coordinate],
                circle_center[coordinate] + amplitude);
        }

        if (periodic_angle_in_interval(
                maximum_angle + M_PI,
                -angular_extent, angular_extent)) {
            box.min[coordinate] = std::min(
                box.min[coordinate],
                circle_center[coordinate] - amplitude);
        }
    }

    return box;
}

std::vector<AABB> build_blue_boxes_rb(const std::vector<Vec3>& positions, const std::vector<Vec3>& x_coms, const std::vector<Vec4>& orientations, const std::vector<Vec4>& quaternion_bounds, const std::vector<double>& prev_com_disp, const SimParams& params, const RefMesh& ref_mesh) {
    const int num_rbs = static_cast<int>(ref_mesh.rb_nodes.size());
    std::vector<AABB> blue_boxes(positions.size());
    constexpr double node_box_padding = 1.2;

    for (int rb = 0; rb < num_rbs; ++rb) {
        const std::vector<int>& nodes = ref_mesh.rb_nodes[rb];
        const std::vector<Vec3>& material_positions = ref_mesh.ref_positions[rb];
        const double r = std::clamp(node_box_padding * prev_com_disp[rb], params.node_box_min, params.node_box_max);
        const Vec3 radius = Vec3::Constant(r);
        for (int local = 0; local < static_cast<int>(nodes.size()); ++local) {
            const int node = nodes[local];
            const AABB rotation_box = arc_node_aabb(x_coms[rb], orientations[rb], material_positions[local], quaternion_bounds[rb]);
            blue_boxes[node] = AABB(rotation_box.min - radius, rotation_box.max + radius);
        }
    }
    return blue_boxes;
}

std::vector<std::vector<int>> build_elastic_adj(const RefMesh& ref_mesh, const VertexTriangleMap& adj, int nv){
    std::vector<std::vector<int>> out(nv);
    #pragma omp parallel for schedule(static)
    for (int vi = 0; vi < nv; ++vi) {
        auto it = adj.find(vi);
        if (it == adj.end()) continue;
        std::vector<int>& row = out[vi];
        for (const auto& [ti, local_a] : it->second) {
            for (int local_b = 0; local_b < 3; ++local_b) {
                const int vj = tri_vertex(ref_mesh, ti, local_b);
                if (vj == vi || vj < 0 || vj >= nv) continue;
                row.push_back(vj);
            }
        }

        // A hinge's two apex vertices (h.v[2], h.v[3]) are coupled through the
        // bending term but share no triangle, so the 1-ring adjacency above
        // misses that pair. The shared-edge endpoints (h.v[0], h.v[1]) are
        // already adjacent to both apexes via the two triangles.
        auto hinge_it = ref_mesh.hinge_adj.find(vi);
        if (hinge_it != ref_mesh.hinge_adj.end()) {
            for (const auto& [hi, role] : hinge_it->second) {
                if (role < 2) continue;
                const Hinge& h = ref_mesh.hinges[hi];
                const int other_apex = h.v[role == 2 ? 3 : 2];
                if (other_apex != vi && other_apex >= 0 && other_apex < nv) row.push_back(other_apex);
            }
        }

        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
    return out;
}

void build_contact_adj(const BroadPhase::Cache& bp_cache, int num_vertices, std::vector<std::vector<int>>& out) {
    // BroadPhase has already created the node-triangle (nt_pairs) and
    // segment-segment (ss_pairs) contact pairs. It also records which contacts
    // contain each vertex in vertex_nt and vertex_ss.
    //
    // No contact search is needed here. For each vertex, read only its cached
    // contact references, fetch the corresponding contact pair, and add the
    // other three vertices as neighbors. The resulting adjacency row is then
    // sorted and deduplicated. Output row capacity is reused between calls.
    if (static_cast<int>(out.size()) == num_vertices) {
        for (auto& neighbors : out) neighbors.clear();
    } else {
        out.assign(num_vertices, {});
    }

    // Each thread owns one output row. For every cached contact containing
    // this vertex, add the other three vertices.
    #pragma omp parallel for schedule(dynamic, 64)
    for (int vertex = 0; vertex < num_vertices; ++vertex) {
        std::vector<int>& neighbors = out[vertex];
        neighbors.reserve(3 * (bp_cache.vertex_nt[vertex].size() + bp_cache.vertex_ss[vertex].size()));

        // Node-triangle contacts containing this vertex.
        for (const auto& cached_nt : bp_cache.vertex_nt[vertex]) {
            if (cached_nt.pair_index >= bp_cache.nt_pairs.size() || cached_nt.dof < 0 || cached_nt.dof >= 4) continue;

            const NodeTrianglePair& contact = bp_cache.nt_pairs[cached_nt.pair_index];
            const int contact_vertices[4] = {
                    contact.node,
                    contact.tri_v[0],
                    contact.tri_v[1],
                    contact.tri_v[2],
            };
            for (int role = 0; role < 4; ++role) {
                if (role == cached_nt.dof) continue;
                const int neighbor = contact_vertices[role];
                if (neighbor >= 0 && neighbor < num_vertices)
                    neighbors.push_back(neighbor);
            }
        }

        // Segment-segment contacts containing this vertex.
        for (const auto& cached_ss : bp_cache.vertex_ss[vertex]) {
            if (cached_ss.pair_index >= bp_cache.ss_pairs.size() || cached_ss.dof < 0 || cached_ss.dof >= 4) continue;

            const SegmentSegmentPair& contact = bp_cache.ss_pairs[cached_ss.pair_index];
            for (int role = 0; role < 4; ++role) {
                if (role == cached_ss.dof) continue;
                const int neighbor = contact.v[role];
                if (neighbor >= 0 && neighbor < num_vertices)
                    neighbors.push_back(neighbor);
            }
        }

        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
}

void union_adjacency(const std::vector<std::vector<int>>& a,const std::vector<std::vector<int>>& b, std::vector<std::vector<int>>& out) {
    const int nv = static_cast<int>(std::max(a.size(), b.size()));
    static const std::vector<int> empty_row;
    if (static_cast<int>(out.size()) == nv) {
        for (auto& row : out) row.clear();
    } else {
        out.assign(nv, {});
    }
    #pragma omp parallel for schedule(dynamic, 64)
    for (int vi = 0; vi < nv; ++vi) {
        const auto& row_a = vi < static_cast<int>(a.size()) ? a[vi] : empty_row;
        const auto& row_b = vi < static_cast<int>(b.size()) ? b[vi] : empty_row;
        out[vi].reserve(row_a.size() + row_b.size());
        std::set_union(row_a.begin(), row_a.end(), row_b.begin(), row_b.end(), std::back_inserter(out[vi]));
    }
}

void greedy_color_conflict_graph(const std::vector<std::vector<int>>& graph, std::vector<std::vector<int>>& groups) {
    const int nv = static_cast<int>(graph.size());
    std::vector<int> color(nv, -1);
    // A timestamped marker array avoids allocating and clearing `used` once per vertex
    // At most `nv` colors can occur in a graph of `nv` vertices.
    std::vector<int> seen_color(nv, -1);
    int max_color = -1;

    for (int vi = 0; vi < nv; ++vi) {
        for (int nb : graph[vi]) {
            if (nb >= 0 && nb < nv && color[nb] >= 0)
                seen_color[color[nb]] = vi;
        }
        int c = 0;
        while (c < nv && seen_color[c] == vi) ++c;
        color[vi] = c;
        max_color = std::max(max_color, c);
    }

    const int num_groups = max_color + 1;
    if (static_cast<int>(groups.size()) == num_groups) {
        for (auto& group : groups) group.clear();
    } else {
        groups.assign(num_groups, {});
    }
    for (int vi = 0; vi < nv; ++vi) {
        if (color[vi] >= 0) groups[color[vi]].push_back(vi);
    }
}
