#include "parallel_helper.h"
#include "quaternion_math.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

int owning_rb_for_node(const std::vector<int>& node_to_rb, int node) {
    return node >= 0 && node < static_cast<int>(node_to_rb.size()) ? node_to_rb[node] : -1;
}

AABB spherical_cap_node_aabb(const Vec3& x_com, const Vec4& q, const Vec3& X, const Vec4& q_rel) {
    if (!x_com.allFinite() || !X.allFinite())
        throw std::invalid_argument("spherical_cap_node_aabb requires finite positions");

    const Vec4 q_current = quaternion_normalize(q);
    const Vec4 q_relative = quaternion_normalize(q_rel);

    const Vec3 world_space_offset = quaternion_rotate(q_current, X);
    const double radius = world_space_offset.norm();
    if (radius == 0.0)
        return AABB(x_com, x_com);

    // Preserve the raw quaternion sign so the angular extent remains in
    // [0, 2*pi] instead of being forced onto the shortest quaternion arc.
    const double sin_half_angle = q_relative.tail<3>().norm();
    const double angular_extent = 2.0 * std::atan2(sin_half_angle, q_relative[0]);

    // With every world-space rotation axis admissible, an extent of pi can
    // reach every direction on the sphere. Larger full-arc extents do too.
    if (angular_extent >= M_PI) {
        const Vec3 sphere_radius = Vec3::Constant(radius);
        return AABB(x_com - sphere_radius, x_com + sphere_radius);
    }

    // For angular_extent < pi, the reachable directions form the spherical
    // cap { y : ||y|| = 1, direction.dot(y) >= cos(angular_extent) }.
    const Vec3 direction = world_space_offset / radius;
    const double cos_extent = std::cos(angular_extent);
    const double sin_extent = std::sin(angular_extent);
    Vec3 cap_min;
    Vec3 cap_max;

    // Optimize each normalized coordinate y_i over the cap. If +/-e_i lies
    // in the cap, it is the corresponding extremum. Otherwise the extremum
    // lies on the cap's boundary circle.
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        const double d = std::clamp(direction[coordinate], -1.0, 1.0);
        const double tangent_length = std::sqrt(std::max(0.0, 1.0 - d * d));

        cap_max[coordinate] = d >= cos_extent
            ? 1.0
            : d * cos_extent + tangent_length * sin_extent;
        cap_min[coordinate] = -d >= cos_extent
            ? -1.0
            : d * cos_extent - tangent_length * sin_extent;
    }

    return AABB(
        x_com + radius * cap_min,
        x_com + radius * cap_max);
}

std::vector<AABB> build_blue_boxes_rb(const std::vector<Vec3>& positions, const std::vector<Vec3>& x_coms, const std::vector<Vec4>& orientations, const std::vector<Vec4>& quaternion_bounds, const std::vector<double>& prev_com_disp, const SimParams& params, const RefMesh& ref_mesh) {
    const int num_rbs = static_cast<int>(ref_mesh.rb_nodes.size());
    std::vector<AABB> blue_boxes(positions.size());
    constexpr double node_box_padding = 1.2;

    for (int rb = 0; rb < num_rbs; ++rb) {
        const std::vector<int>& nodes = ref_mesh.rb_nodes[rb];
        const std::vector<Vec3>& material_positions = ref_mesh.ref_positions[rb];
        const Vec3 com_radius = Vec3::Constant(std::clamp(node_box_padding * prev_com_disp[rb], params.node_box_min, params.node_box_max));
        for (int local = 0; local < static_cast<int>(nodes.size()); ++local) {
            const int node = nodes[local];
            const AABB spherical_cap_box = spherical_cap_node_aabb(x_coms[rb], orientations[rb], material_positions[local], quaternion_bounds[rb]);
            blue_boxes[node] = AABB(spherical_cap_box.min - com_radius, spherical_cap_box.max + com_radius);
        }
    }
    return blue_boxes;
}

void build_rb_contact_adj(const BroadPhase::Cache& bp_cache, const std::vector<int>& node_to_rb, int num_rbs, std::vector<std::vector<int>>& out) {
    if (static_cast<int>(out.size()) == num_rbs) {
        for (std::vector<int>& neighbors : out)
            neighbors.clear();
    } else {
        out.assign(num_rbs, {});
    }

    const auto add_edge = [&](int first, int second) {
        if (first < 0 || second < 0)
            return;
        out[first].push_back(second);
        out[second].push_back(first);
    };

    for (const NodeTrianglePair& pair : bp_cache.nt_pairs) {
        const int node_rb = owning_rb_for_node(node_to_rb, pair.node);
        const int triangle_rb = owning_rb_for_node(node_to_rb, pair.tri_v[0]);
        if (node_rb == triangle_rb || (node_rb < 0 && triangle_rb < 0))
            continue;
        add_edge(node_rb, triangle_rb);
    }

    for (const SegmentSegmentPair& pair : bp_cache.ss_pairs) {
        const int first_edge_rb = owning_rb_for_node(node_to_rb, pair.v[0]);
        const int second_edge_rb = owning_rb_for_node(node_to_rb, pair.v[2]);
        if (first_edge_rb == second_edge_rb || (first_edge_rb < 0 && second_edge_rb < 0))
            continue;
        add_edge(first_edge_rb, second_edge_rb);
    }

    for (std::vector<int>& neighbors : out) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
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
