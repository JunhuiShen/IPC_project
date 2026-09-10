#include "contact_scheduling.h"
#include "parallel_helper.h"
#include "quaternion_math.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <exception>
#include <omp.h>

int owning_rb_for_node(const std::vector<int>& node_to_rb, int node) {
    return node >= 0 && node < static_cast<int>(node_to_rb.size()) ? node_to_rb[node] : -1;
}

AABB spherical_cap_node_aabb(const Vec3& x_com, const Vec4& q, const Vec3& X, double theta_bound) {
    if (!x_com.allFinite() || !X.allFinite())
        throw std::invalid_argument("spherical_cap_node_aabb requires finite positions");

    const Vec4 q_current = quaternion_normalize(q);
    const Vec3 world_space_offset = quaternion_rotate(q_current, X);
    const double radius = world_space_offset.norm();
    if (radius == 0.0)
        return AABB(x_com, x_com);

    const double angular_extent = std::max(0.0, theta_bound);
    if (angular_extent >= M_PI) {
        const Vec3 sphere_radius = Vec3::Constant(radius);
        return AABB(x_com - sphere_radius, x_com + sphere_radius);
    }

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

    return AABB(x_com + radius * cap_min, x_com + radius * cap_max);
}

void build_blue_boxes_rb(const std::vector<Vec3>& com_box_anchors, const std::vector<Vec4>& orientation_box_anchors, const std::vector<double>& theta_box_radii, const std::vector<double>& com_box_radii, const RefMesh& ref_mesh, std::vector<AABB>& blue_boxes) {
    const int num_rbs = static_cast<int>(ref_mesh.rb_nodes.size());

    std::vector<std::size_t> offsets(static_cast<std::size_t>(num_rbs) + 1, 0);
    for (int rb = 0; rb < num_rbs; ++rb)
        offsets[rb + 1] = offsets[rb] + ref_mesh.rb_nodes[rb].size();
    std::exception_ptr failure;
    // Flatten the proxy nodes so even a single large body uses the team.
    #pragma omp parallel if(offsets.back() >= 128)
    {
        const std::size_t begin = offsets.back() * omp_get_thread_num() / omp_get_num_threads();
        const std::size_t end = offsets.back() * (omp_get_thread_num() + 1) / omp_get_num_threads();
        try {
            int rb = static_cast<int>(std::upper_bound(offsets.begin(), offsets.end(), begin) - offsets.begin()) - 1;
            for (std::size_t flat = begin; flat < end; ++flat) {
                while (flat >= offsets[rb + 1]) ++rb;
                const std::size_t local = flat - offsets[rb];
                const int node = ref_mesh.rb_nodes[rb][local];
                const Vec3 com_radius = Vec3::Constant(com_box_radii[rb]);
                const AABB spherical_cap_box = spherical_cap_node_aabb(com_box_anchors[rb], orientation_box_anchors[rb], ref_mesh.ref_positions[rb][local], theta_box_radii[rb]);
                blue_boxes[node] = AABB(spherical_cap_box.min - com_radius, spherical_cap_box.max + com_radius);
            }
        } catch (...) {
            #pragma omp critical(rigid_blue_box_exception)
            { if (!failure) failure = std::current_exception(); }
        }
    }
    if (failure) std::rethrow_exception(failure);
}

void build_rb_contact_adj(const BroadPhase::Cache& bp_cache, const std::vector<int>& node_to_rb, int num_rbs, std::vector<std::vector<int>>& body_nt_pair_indices, std::vector<std::vector<int>>& body_ss_pair_indices, std::vector<std::vector<int>>& out) {
    out.resize(static_cast<std::size_t>(num_rbs));
    body_nt_pair_indices.resize(static_cast<std::size_t>(num_rbs));
    body_ss_pair_indices.resize(static_cast<std::size_t>(num_rbs));

    // Each worker owns counts and output intervals for a contiguous range of
    // pairs. Concatenating worker intervals preserves ascending pair order.
    const auto gather_pairs = [&](const auto& pairs, const auto& owners, std::vector<std::vector<int>>& body_pairs) {
        const int workers = pairs.size() >= 128 ? omp_get_max_threads() : 1;
        std::vector<std::size_t> offsets(static_cast<std::size_t>(workers) * num_rbs, 0);
        #pragma omp parallel for schedule(static, 1) if(workers > 1)
        for (int worker = 0; worker < workers; ++worker) {
            const std::size_t begin = pairs.size() * worker / workers;
            const std::size_t end = pairs.size() * (worker + 1) / workers;
            for (std::size_t i = begin; i < end; ++i) {
                const auto [first, second] = owners(pairs[i]);
                if (first == second) continue;
                if (first >= 0) ++offsets[static_cast<std::size_t>(worker) * num_rbs + first];
                if (second >= 0) ++offsets[static_cast<std::size_t>(worker) * num_rbs + second];
            }
        }
        #pragma omp parallel for schedule(static) if(num_rbs >= 8)
        for (int rb = 0; rb < num_rbs; ++rb) {
            std::size_t offset = 0;
            for (int worker = 0; worker < workers; ++worker) {
                std::size_t& slot = offsets[static_cast<std::size_t>(worker) * num_rbs + rb];
                const std::size_t count = slot;
                slot = offset;
                offset += count;
            }
            body_pairs[rb].resize(offset);
        }
        #pragma omp parallel for schedule(static, 1) if(workers > 1)
        for (int worker = 0; worker < workers; ++worker) {
            const std::size_t begin = pairs.size() * worker / workers;
            const std::size_t end = pairs.size() * (worker + 1) / workers;
            for (std::size_t i = begin; i < end; ++i) {
                const auto [first, second] = owners(pairs[i]);
                if (first == second) continue;
                if (first >= 0) body_pairs[first][offsets[static_cast<std::size_t>(worker) * num_rbs + first]++] = static_cast<int>(i);
                if (second >= 0) body_pairs[second][offsets[static_cast<std::size_t>(worker) * num_rbs + second]++] = static_cast<int>(i);
            }
        }
    };
    const auto nt_owners = [&](const NodeTrianglePair& pair) {
        return std::pair<int, int>{owning_rb_for_node(node_to_rb, pair.node),
            owning_rb_for_node(node_to_rb, pair.tri_v[0])};
    };
    const auto ss_owners = [&](const SegmentSegmentPair& pair) {
        return std::pair<int, int>{owning_rb_for_node(node_to_rb, pair.v[0]),
            owning_rb_for_node(node_to_rb, pair.v[2])};
    };
    gather_pairs(bp_cache.nt_pairs, nt_owners, body_nt_pair_indices);
    gather_pairs(bp_cache.ss_pairs, ss_owners, body_ss_pair_indices);
    #pragma omp parallel for schedule(dynamic, 1) if(num_rbs >= 8)
    for (int rb = 0; rb < num_rbs; ++rb) {
        auto& neighbors = out[rb];
        neighbors.clear();
        const auto append_other = [&](const std::pair<int, int>& owners) {
            const int other = owners.first == rb ? owners.second : owners.first;
            if (other >= 0) neighbors.push_back(other);
        };
        for (const int i : body_nt_pair_indices[rb]) append_other(nt_owners(bp_cache.nt_pairs[i]));
        for (const int i : body_ss_pair_indices[rb]) append_other(ss_owners(bp_cache.ss_pairs[i]));
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
}

std::vector<int> build_node_to_block(
    const std::vector<int>& node_to_rb,
    const std::vector<int>& deformable_nodes,
    int num_rbs) {
    if (num_rbs < 0)
        throw std::invalid_argument("mixed adjacency: negative rigid-body count");

    const int num_nodes = static_cast<int>(node_to_rb.size());
    const int num_deformable = static_cast<int>(deformable_nodes.size());
    std::vector<int> node_to_block(num_nodes, -1);

    for (int block = 0; block < num_deformable; ++block) {
        const int node = deformable_nodes[block];
        if (node < 0 || node >= num_nodes)
            throw std::out_of_range("mixed adjacency: deformable node is out of range");
        if (node_to_rb[node] != -1)
            throw std::invalid_argument("mixed adjacency: deformable node has a rigid owner");
        if (node_to_block[node] >= 0)
            throw std::invalid_argument("mixed adjacency: duplicate deformable node");
        node_to_block[node] = block;
    }

    for (int node = 0; node < num_nodes; ++node) {
        const int rb = node_to_rb[node];
        if (rb < -1 || rb >= num_rbs)
            throw std::invalid_argument("mixed adjacency: invalid rigid-body owner");
        if (rb >= 0)
            node_to_block[node] = num_deformable + rb;
        else if (node_to_block[node] < 0)
            throw std::invalid_argument("mixed adjacency: deformable-node list is incomplete");
    }

    return node_to_block;
}

void build_block_elastic_adj(
    const std::vector<std::vector<int>>& nodal_elastic_adj,
    const std::vector<int>& node_to_block,
    const std::vector<std::vector<int>>& block_nodes,
    std::vector<std::vector<int>>& out) {
    const int num_nodes = static_cast<int>(node_to_block.size());
    const int num_blocks = static_cast<int>(block_nodes.size());

    out.resize(num_blocks);

    #pragma omp parallel for schedule(dynamic, 16)
    for (int block = 0; block < num_blocks; ++block) {
        std::vector<int>& row = out[block];
        row.clear();
        for (const int node : block_nodes[block]) {
            for (const int neighbor : nodal_elastic_adj[node]) {
                const int neighbor_block = node_to_block[neighbor];
                if (neighbor_block != block)
                    row.push_back(neighbor_block);
            }
        }
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
}

void build_block_contact_adj(const BroadPhase::Cache& bp_cache, const std::vector<int>& node_to_block, const std::vector<std::vector<int>>& block_nodes, int num_deformable_blocks,
    std::vector<std::vector<int>>& body_nt_pair_indices, std::vector<std::vector<int>>& body_ss_pair_indices, std::vector<std::vector<int>>& out) {
    const int num_blocks = static_cast<int>(block_nodes.size());
    const int num_rbs = num_blocks - num_deformable_blocks;

    out.resize(num_blocks);
    body_nt_pair_indices.resize(num_rbs);
    body_ss_pair_indices.resize(num_rbs);

    #pragma omp parallel for schedule(dynamic, 16)
    for (int block = 0; block < num_blocks; ++block) {
        std::vector<int>& neighbors = out[block];
        neighbors.clear();
        std::vector<int>* rigid_nt = nullptr;
        std::vector<int>* rigid_ss = nullptr;
        if (block >= num_deformable_blocks) {
            const int rb = block - num_deformable_blocks;
            rigid_nt = &body_nt_pair_indices[rb];
            rigid_ss = &body_ss_pair_indices[rb];
            rigid_nt->clear();
            rigid_ss->clear();
        }

        std::size_t incidence_count = 0;
        for (const int node : block_nodes[block]) {
            incidence_count += bp_cache.vertex_nt[node].size();
            incidence_count += bp_cache.vertex_ss[node].size();
        }
        neighbors.reserve(3 * incidence_count);
        if (rigid_nt)
            rigid_nt->reserve(incidence_count);
        if (rigid_ss)
            rigid_ss->reserve(incidence_count);

        const auto add_neighbor_blocks = [&](const int nodes[4]) {
            for (int role = 0; role < 4; ++role) {
                const int neighbor_block = node_to_block[nodes[role]];
                if (neighbor_block != block)
                    neighbors.push_back(neighbor_block);
            }
        };

        for (const int node : block_nodes[block]) {
            for (const BroadPhase::Cache::VertexPairEntry& entry :
                 bp_cache.vertex_nt[node]) {
                const NodeTrianglePair& pair = bp_cache.nt_pairs[entry.pair_index];
                const int nodes[4] = {pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]};
                int representative_role = 0;
                while (representative_role < 4 && node_to_block[nodes[representative_role]] != block)
                    ++representative_role;
                if (entry.dof != representative_role)
                    continue;
                add_neighbor_blocks(nodes);
                if (rigid_nt) {
                    const int first = node_to_block[pair.node];
                    const int second = node_to_block[pair.tri_v[0]];
                    if (first != second && (block == first || block == second))
                        rigid_nt->push_back(static_cast<int>(entry.pair_index));
                }
            }
            for (const BroadPhase::Cache::VertexPairEntry& entry :
                 bp_cache.vertex_ss[node]) {
                const SegmentSegmentPair& pair = bp_cache.ss_pairs[entry.pair_index];
                int representative_role = 0;
                while (representative_role < 4 && node_to_block[pair.v[representative_role]] != block)
                    ++representative_role;
                if (entry.dof != representative_role)
                    continue;
                add_neighbor_blocks(pair.v);
                if (rigid_ss) {
                    const int first = node_to_block[pair.v[0]];
                    const int second = node_to_block[pair.v[2]];
                    if (first != second && (block == first || block == second))
                        rigid_ss->push_back(static_cast<int>(entry.pair_index));
                }
            }
        }

        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(
            std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        if (rigid_nt) {
            std::sort(rigid_nt->begin(), rigid_nt->end());
            rigid_nt->erase(
                std::unique(rigid_nt->begin(), rigid_nt->end()),
                rigid_nt->end());
        }
        if (rigid_ss) {
            std::sort(rigid_ss->begin(), rigid_ss->end());
            rigid_ss->erase(
                std::unique(rigid_ss->begin(), rigid_ss->end()),
                rigid_ss->end());
        }
    }
}

std::vector<std::vector<int>> build_solid_elastic_adjacency(
    const RefMesh& ref_mesh) {
    const int num_solid_nodes =
        static_cast<int>(ref_mesh.tet_nodes.size());
    std::vector<int> node_to_solid(ref_mesh.num_positions, -1);
    #pragma omp parallel for schedule(static) if(num_solid_nodes >= 128)
    for (int solid = 0; solid < num_solid_nodes; ++solid) {
        const int node = ref_mesh.tet_nodes[static_cast<std::size_t>(solid)];
        node_to_solid[static_cast<std::size_t>(node)] = solid;
    }

    std::vector<std::vector<int>> adjacency(
        static_cast<std::size_t>(num_solid_nodes));

    if (ref_mesh.tet_adj.size() == static_cast<std::size_t>(ref_mesh.num_positions)) {
        #pragma omp parallel for schedule(dynamic, 16) if(num_solid_nodes >= 128)
        for (int solid = 0; solid < num_solid_nodes; ++solid) {
            const int node = ref_mesh.tet_nodes[solid];
            auto& row = adjacency[solid];
            for (const auto& [element, local_role] : ref_mesh.tet_adj[node]) {
                for (int role = 0; role < 4; ++role) {
                    const int neighbor = node_to_solid[tet_vertex(ref_mesh, element, role)];
                    if (neighbor >= 0 && neighbor != solid) row.push_back(neighbor);
                }
            }
        }
    } else {
        // Raw connectivity callers may not have built tet incidence yet.
        for (int element = 0; element < num_tets(ref_mesh); ++element) {
            int solid_nodes[4];
            for (int local = 0; local < 4; ++local) {
                const int node = tet_vertex(ref_mesh, element, local);
                solid_nodes[local] =
                    node_to_solid[static_cast<std::size_t>(node)];
            }
            for (int first = 0; first < 4; ++first) {
                for (int second = first + 1; second < 4; ++second) {
                    const int a = solid_nodes[first];
                    const int b = solid_nodes[second];
                    if (a < 0 || b < 0 || a == b)
                        continue;
                    adjacency[static_cast<std::size_t>(a)].push_back(b);
                    adjacency[static_cast<std::size_t>(b)].push_back(a);
                }
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (int solid = 0; solid < num_solid_nodes; ++solid) {
        std::vector<int>& neighbors =
            adjacency[static_cast<std::size_t>(solid)];
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(
            std::unique(neighbors.begin(), neighbors.end()),
            neighbors.end());
    }
    return adjacency;
}

void build_solid_contact_adjacency(
    const RefMesh& ref_mesh,
    const BroadPhase::Cache& bp_cache,
    std::vector<std::vector<int>>& out) {
    const int num_solid_nodes = static_cast<int>(ref_mesh.tet_nodes.size());
    std::vector<int> node_to_solid(ref_mesh.num_positions, -1);
    #pragma omp parallel for schedule(static) if(num_solid_nodes >= 128)
    for (int solid = 0; solid < num_solid_nodes; ++solid) {
        const int node = ref_mesh.tet_nodes[static_cast<std::size_t>(solid)];
        node_to_solid[static_cast<std::size_t>(node)] = solid;
    }
    std::vector<unsigned char> surface_node(ref_mesh.num_positions, 0);
    #pragma omp parallel for schedule(static) if(ref_mesh.surface_nodes.size() >= 128)
    for (int surface = 0; surface < static_cast<int>(ref_mesh.surface_nodes.size()); ++surface)
        surface_node[static_cast<std::size_t>(ref_mesh.surface_nodes[surface])] = 1;

    out.resize(static_cast<std::size_t>(num_solid_nodes));

    #pragma omp parallel for schedule(dynamic, 64)
    for (int solid = 0; solid < num_solid_nodes; ++solid) {
        out[solid].clear();
        const int node = ref_mesh.tet_nodes[static_cast<std::size_t>(solid)];
        std::vector<int>& neighbors = out[static_cast<std::size_t>(solid)];
        neighbors.reserve(
            3 * (bp_cache.vertex_nt[static_cast<std::size_t>(node)].size()
                 + bp_cache.vertex_ss[static_cast<std::size_t>(node)].size()));

        for (const BroadPhase::Cache::VertexPairEntry& entry :
             bp_cache.vertex_nt[static_cast<std::size_t>(node)]) {
            const NodeTrianglePair& pair = bp_cache.nt_pairs[entry.pair_index];
            const int contact_nodes[4] = {
                pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]};

            bool contains_interior_solid_node = false;
            for (const int contact_node : contact_nodes) {
                const int contact_solid = node_to_solid[
                    static_cast<std::size_t>(contact_node)];
                if (contact_solid >= 0
                    && surface_node[static_cast<std::size_t>(contact_node)]
                        == 0) {
                    contains_interior_solid_node = true;
                    break;
                }
            }
            if (contains_interior_solid_node)
                continue;

            for (const int contact_node : contact_nodes) {
                const int neighbor = node_to_solid[
                    static_cast<std::size_t>(contact_node)];
                if (neighbor >= 0 && neighbor != solid)
                    neighbors.push_back(neighbor);
            }
        }

        for (const BroadPhase::Cache::VertexPairEntry& entry :
             bp_cache.vertex_ss[static_cast<std::size_t>(node)]) {
            const SegmentSegmentPair& pair =
                bp_cache.ss_pairs[entry.pair_index];

            bool contains_interior_solid_node = false;
            for (const int contact_node : pair.v) {
                const int contact_solid = node_to_solid[
                    static_cast<std::size_t>(contact_node)];
                if (contact_solid >= 0
                    && surface_node[static_cast<std::size_t>(contact_node)]
                        == 0) {
                    contains_interior_solid_node = true;
                    break;
                }
            }
            if (contains_interior_solid_node)
                continue;

            for (const int contact_node : pair.v) {
                const int neighbor = node_to_solid[
                    static_cast<std::size_t>(contact_node)];
                if (neighbor >= 0 && neighbor != solid)
                    neighbors.push_back(neighbor);
            }
        }

        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(
            std::unique(neighbors.begin(), neighbors.end()),
            neighbors.end());
    }
}

void build_all_block_adjacency_and_contact(const RefMesh& ref_mesh, const std::vector<int>& cloth_nodes, const std::vector<std::vector<int>>& cloth_nodal_elastic_adjacency, const BroadPhase::Cache& bp_cache, std::vector<std::vector<int>>& out, const std::vector<int>* node_to_block, const std::vector<unsigned char>* solid_node_mask, const std::vector<unsigned char>* surface_node_mask, std::vector<std::size_t>* elastic_row_sizes, const std::vector<std::vector<int>>* body_nt_pair_indices, const std::vector<std::vector<int>>* body_ss_pair_indices) {
    const int num_cloth = static_cast<int>(cloth_nodes.size());
    const int num_solid = static_cast<int>(ref_mesh.tet_nodes.size());
    const int num_rigid = static_cast<int>(ref_mesh.rb_nodes.size());
    const int solid_begin = num_cloth;
    const int rigid_begin = solid_begin + num_solid;
    const int num_blocks = rigid_begin + num_rigid;

    const int num_workspace_arguments = static_cast<int>(node_to_block != nullptr) + static_cast<int>(solid_node_mask != nullptr) + static_cast<int>(surface_node_mask != nullptr) + static_cast<int>(elastic_row_sizes != nullptr);
    if (num_workspace_arguments != 0 && num_workspace_arguments != 4)
        throw std::invalid_argument("build_all_block_adjacency_and_contact: supply either all workspace arguments or none");
    if ((body_nt_pair_indices == nullptr) != (body_ss_pair_indices == nullptr)) throw std::invalid_argument("build_all_block_adjacency_and_contact: supply both rigid pair lists or neither");
    if (body_nt_pair_indices != nullptr && (body_nt_pair_indices->size() != static_cast<std::size_t>(num_rigid) || body_ss_pair_indices->size() != static_cast<std::size_t>(num_rigid))) throw std::invalid_argument("build_all_block_adjacency_and_contact: rigid pair-list size mismatch");

    std::vector<int> owned_node_to_block;
    std::vector<unsigned char> owned_solid_node_mask;
    std::vector<unsigned char> owned_surface_node_mask;
    std::vector<std::size_t> owned_elastic_row_sizes;
    if (node_to_block == nullptr) {
        owned_node_to_block.assign(ref_mesh.num_positions, -1);
        owned_solid_node_mask.assign(ref_mesh.num_positions, 0);
        owned_surface_node_mask.assign(ref_mesh.num_positions, 0);
        for (int cloth = 0; cloth < num_cloth; ++cloth)
            owned_node_to_block[static_cast<std::size_t>(cloth_nodes[static_cast<std::size_t>(cloth)])] = cloth;
        for (int solid = 0; solid < num_solid; ++solid) {
            const int node = ref_mesh.tet_nodes[static_cast<std::size_t>(solid)];
            owned_node_to_block[static_cast<std::size_t>(node)] = solid_begin + solid;
            owned_solid_node_mask[static_cast<std::size_t>(node)] = 1;
        }
        for (const int node : ref_mesh.surface_nodes)
            owned_surface_node_mask[static_cast<std::size_t>(node)] = 1;
        for (int rigid = 0; rigid < num_rigid; ++rigid) {
            for (const int node : ref_mesh.rb_nodes[static_cast<std::size_t>(rigid)])
                owned_node_to_block[static_cast<std::size_t>(node)] = rigid_begin + rigid;
        }
        node_to_block = &owned_node_to_block;
        solid_node_mask = &owned_solid_node_mask;
        surface_node_mask = &owned_surface_node_mask;
        elastic_row_sizes = &owned_elastic_row_sizes;
    }

    const bool rebuild_elastic = out.size() != static_cast<std::size_t>(num_blocks) || elastic_row_sizes->size() != static_cast<std::size_t>(num_blocks);
    if (rebuild_elastic) {
        out.assign(static_cast<std::size_t>(num_blocks), {});

        #pragma omp parallel for schedule(static) if(num_cloth >= 128)
        for (int cloth = 0; cloth < num_cloth; ++cloth) {
            const int node = cloth_nodes[static_cast<std::size_t>(cloth)];
            std::vector<int>& row = out[static_cast<std::size_t>(cloth)];
            for (const int neighbor_node : cloth_nodal_elastic_adjacency[static_cast<std::size_t>(node)]) {
                const int neighbor = (*node_to_block)[static_cast<std::size_t>(neighbor_node)];
                if (neighbor >= 0 && neighbor < num_cloth && neighbor != cloth)
                    row.push_back(neighbor);
            }
        }

        if (ref_mesh.tet_adj.size() == static_cast<std::size_t>(ref_mesh.num_positions)) {
            #pragma omp parallel for schedule(dynamic, 16) if(num_solid >= 128)
            for (int solid = 0; solid < num_solid; ++solid) {
                const int node = ref_mesh.tet_nodes[solid];
                const int block = solid_begin + solid;
                auto& row = out[block];
                for (const auto& [element, local_role] : ref_mesh.tet_adj[node]) {
                    for (int role = 0; role < 4; ++role) {
                        const int neighbor = (*node_to_block)[tet_vertex(ref_mesh, element, role)];
                        if (neighbor >= 0 && neighbor != block) row.push_back(neighbor);
                    }
                }
            }
        } else {
            for (int element = 0; element < num_tets(ref_mesh); ++element) {
                int blocks[4];
                for (int local = 0; local < 4; ++local)
                    blocks[local] = (*node_to_block)[static_cast<std::size_t>(tet_vertex(ref_mesh, element, local))];
                for (int first = 0; first < 4; ++first) {
                    for (int second = first + 1; second < 4; ++second) {
                        const int a = blocks[first];
                        const int b = blocks[second];
                        if (a < 0 || b < 0 || a == b)
                            continue;
                        out[static_cast<std::size_t>(a)].push_back(b);
                        out[static_cast<std::size_t>(b)].push_back(a);
                    }
                }
            }
        }

#pragma omp parallel for schedule(static)
        for (int block = 0; block < num_blocks; ++block) {
            std::vector<int>& row = out[static_cast<std::size_t>(block)];
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
        }
        elastic_row_sizes->resize(static_cast<std::size_t>(num_blocks));
        #pragma omp parallel for schedule(static) if(num_blocks >= 128)
        for (int block = 0; block < num_blocks; ++block)
            (*elastic_row_sizes)[static_cast<std::size_t>(block)] = out[static_cast<std::size_t>(block)].size();
    } else {
        int missing_prefix = 0;
        #pragma omp parallel for schedule(static) reduction(|:missing_prefix) if(num_blocks >= 128)
        for (int block = 0; block < num_blocks; ++block) {
            const std::size_t elastic_size = (*elastic_row_sizes)[static_cast<std::size_t>(block)];
            if (out[static_cast<std::size_t>(block)].size() < elastic_size) missing_prefix = 1;
            else out[static_cast<std::size_t>(block)].resize(elastic_size);
        }
        if (missing_prefix)
            throw std::invalid_argument("build_all_block_adjacency_and_contact: missing elastic prefix");
    }

    const bool has_vertex_incidence = bp_cache.vertex_nt.size() >= node_to_block->size() && bp_cache.vertex_ss.size() >= node_to_block->size();
    std::vector<std::vector<int>> owned_body_nt_pair_indices;
    std::vector<std::vector<int>> owned_body_ss_pair_indices;
    if (has_vertex_incidence && num_rigid > 0 && body_nt_pair_indices == nullptr) {
        owned_body_nt_pair_indices.assign(static_cast<std::size_t>(num_rigid), {});
        owned_body_ss_pair_indices.assign(static_cast<std::size_t>(num_rigid), {});
        const auto append_rigid_pair = [&](const int first_rigid, const int second_rigid, const int pair_index, std::vector<std::vector<int>>& pair_indices) {
            if (first_rigid == second_rigid || (first_rigid < 0 && second_rigid < 0)) return;
            if (first_rigid >= 0) pair_indices[static_cast<std::size_t>(first_rigid)].push_back(pair_index);
            if (second_rigid >= 0) pair_indices[static_cast<std::size_t>(second_rigid)].push_back(pair_index);
        };
        for (int pair_index = 0; pair_index < static_cast<int>(bp_cache.nt_pairs.size()); ++pair_index) {
            const NodeTrianglePair& pair = bp_cache.nt_pairs[static_cast<std::size_t>(pair_index)];
            append_rigid_pair(owning_rb_for_node(ref_mesh.node_to_rb, pair.node), owning_rb_for_node(ref_mesh.node_to_rb, pair.tri_v[0]), pair_index, owned_body_nt_pair_indices);
        }
        for (int pair_index = 0; pair_index < static_cast<int>(bp_cache.ss_pairs.size()); ++pair_index) {
            const SegmentSegmentPair& pair = bp_cache.ss_pairs[static_cast<std::size_t>(pair_index)];
            append_rigid_pair(owning_rb_for_node(ref_mesh.node_to_rb, pair.v[0]), owning_rb_for_node(ref_mesh.node_to_rb, pair.v[2]), pair_index, owned_body_ss_pair_indices);
        }
        body_nt_pair_indices = &owned_body_nt_pair_indices;
        body_ss_pair_indices = &owned_body_ss_pair_indices;
    }

    if (!has_vertex_incidence) {
        const auto append_contact_clique = [&](const int nodes[4]) {
            int blocks[4];
            int num_contact_blocks = 0;
            for (int role = 0; role < 4; ++role) {
                const int node = nodes[role];
                if (!bp_cache.excludes_tet_interior_nt_queries && (*solid_node_mask)[static_cast<std::size_t>(node)] != 0 && (*surface_node_mask)[static_cast<std::size_t>(node)] == 0) return;
                const int block = (*node_to_block)[static_cast<std::size_t>(node)];
                if (block < 0) continue;
                bool duplicate = false;
                for (int existing = 0; existing < num_contact_blocks; ++existing) duplicate = duplicate || blocks[existing] == block;
                if (!duplicate) blocks[num_contact_blocks++] = block;
            }
            for (int first = 0; first < num_contact_blocks; ++first) {
                for (int second = first + 1; second < num_contact_blocks; ++second) {
                    const int lower = std::min(blocks[first], blocks[second]);
                    const int upper = std::max(blocks[first], blocks[second]);
                    out[static_cast<std::size_t>(upper)].push_back(lower);
                }
            }
        };
        for (const NodeTrianglePair& pair : bp_cache.nt_pairs) {
            const int nodes[4] = {pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]};
            append_contact_clique(nodes);
        }
        for (const SegmentSegmentPair& pair : bp_cache.ss_pairs) append_contact_clique(pair.v);
        #pragma omp parallel for schedule(static)
        for (int block = 0; block < num_blocks; ++block) {
            std::vector<int>& row = out[static_cast<std::size_t>(block)];
            const std::size_t elastic_size = (*elastic_row_sizes)[static_cast<std::size_t>(block)];
            auto contact_begin = row.begin() + static_cast<std::ptrdiff_t>(elastic_size);
            std::sort(contact_begin, row.end());
            auto contact_end = std::unique(contact_begin, row.end());
            contact_end = std::remove_if(contact_begin, contact_end, [&](const int neighbor) { return std::binary_search(row.begin(), contact_begin, neighbor); });
            row.erase(contact_end, row.end());
        }
        return;
    }

    #pragma omp parallel for schedule(dynamic, 32)
    for (int block = 0; block < num_blocks; ++block) {
        std::vector<int>& row = out[static_cast<std::size_t>(block)];
        const std::size_t elastic_size = (*elastic_row_sizes)[static_cast<std::size_t>(block)];
        thread_local std::vector<std::size_t> seen_generation;
        thread_local std::size_t generation = 0;
        if (seen_generation.size() != static_cast<std::size_t>(num_blocks)) seen_generation.assign(static_cast<std::size_t>(num_blocks), 0);
        if (++generation == 0) {
            std::fill(seen_generation.begin(), seen_generation.end(), 0);
            ++generation;
        }
        const auto append_contact = [&](const int nodes[4]) {
            if (!bp_cache.excludes_tet_interior_nt_queries) {
                for (int role = 0; role < 4; ++role) {
                    const int node = nodes[role];
                    if ((*solid_node_mask)[static_cast<std::size_t>(node)] != 0 && (*surface_node_mask)[static_cast<std::size_t>(node)] == 0) return;
                }
            }
            for (int role = 0; role < 4; ++role) {
                const int neighbor = (*node_to_block)[static_cast<std::size_t>(nodes[role])];
                if (neighbor >= 0 && neighbor < block && seen_generation[static_cast<std::size_t>(neighbor)] != generation) {
                    seen_generation[static_cast<std::size_t>(neighbor)] = generation;
                    row.push_back(neighbor);
                }
            }
        };
        const auto append_nt_pair = [&](const std::size_t pair_index) {
            const NodeTrianglePair& pair = bp_cache.nt_pairs[pair_index];
            const int nodes[4] = {pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]};
            append_contact(nodes);
        };
        const auto append_ss_pair = [&](const std::size_t pair_index) { append_contact(bp_cache.ss_pairs[pair_index].v); };

        if (block < rigid_begin) {
            const int node = block < num_cloth ? cloth_nodes[static_cast<std::size_t>(block)] : ref_mesh.tet_nodes[static_cast<std::size_t>(block - num_cloth)];
            const std::size_t nt_count = bp_cache.vertex_nt[static_cast<std::size_t>(node)].size();
            const std::size_t ss_count = bp_cache.vertex_ss[static_cast<std::size_t>(node)].size();
            row.reserve(elastic_size + 3 * (nt_count + ss_count));
            for (const BroadPhase::Cache::VertexPairEntry& entry : bp_cache.vertex_nt[static_cast<std::size_t>(node)]) append_nt_pair(entry.pair_index);
            for (const BroadPhase::Cache::VertexPairEntry& entry : bp_cache.vertex_ss[static_cast<std::size_t>(node)]) append_ss_pair(entry.pair_index);
        } else {
            const int rigid = block - rigid_begin;
            const std::vector<int>& nt_indices = (*body_nt_pair_indices)[static_cast<std::size_t>(rigid)];
            const std::vector<int>& ss_indices = (*body_ss_pair_indices)[static_cast<std::size_t>(rigid)];
            row.reserve(elastic_size + 3 * (nt_indices.size() + ss_indices.size()));
            for (const int pair_index : nt_indices) append_nt_pair(static_cast<std::size_t>(pair_index));
            for (const int pair_index : ss_indices) append_ss_pair(static_cast<std::size_t>(pair_index));
        }
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
    // No contact search is needed here. Each contact edge is stored only in
    // its larger-index row, which is sufficient for ascending greedy coloring.
    out.resize(num_vertices);

    // Each thread owns one output row.
    #pragma omp parallel for schedule(dynamic, 64)
    for (int vertex = 0; vertex < num_vertices; ++vertex) {
        std::vector<int>& neighbors = out[vertex];
        neighbors.clear();
        neighbors.reserve(3 * (bp_cache.vertex_nt[vertex].size() + bp_cache.vertex_ss[vertex].size()));
        thread_local std::vector<std::size_t> seen_generation;
        thread_local std::size_t generation = 0;
        if (seen_generation.size() != static_cast<std::size_t>(num_vertices)) seen_generation.assign(static_cast<std::size_t>(num_vertices), 0);
        if (++generation == 0) {
            std::fill(seen_generation.begin(), seen_generation.end(), 0);
            ++generation;
        }
        // Deduplicate before sorting so repeated candidate cliques do not inflate the row or sort workload.
        const auto append_neighbor = [&](const int neighbor) {
            if (neighbor >= 0 && neighbor < vertex && seen_generation[static_cast<std::size_t>(neighbor)] != generation) {
                seen_generation[static_cast<std::size_t>(neighbor)] = generation;
                neighbors.push_back(neighbor);
            }
        };

        // Node-triangle contacts containing this vertex.
        for (const auto& cached_nt : bp_cache.vertex_nt[vertex]) {
            if (cached_nt.pair_index >= bp_cache.nt_pairs.size() || cached_nt.dof < 0 || cached_nt.dof >= 4) continue;

            const NodeTrianglePair& contact = bp_cache.nt_pairs[cached_nt.pair_index];
            const int contact_vertices[4] = {contact.node, contact.tri_v[0], contact.tri_v[1], contact.tri_v[2]};
            for (int role = 0; role < 4; ++role) {
                if (role == cached_nt.dof) continue;
                append_neighbor(contact_vertices[role]);
            }
        }

        // Segment-segment contacts containing this vertex.
        for (const auto& cached_ss : bp_cache.vertex_ss[vertex]) {
            if (cached_ss.pair_index >= bp_cache.ss_pairs.size() || cached_ss.dof < 0 || cached_ss.dof >= 4) continue;

            const SegmentSegmentPair& contact = bp_cache.ss_pairs[cached_ss.pair_index];
            for (int role = 0; role < 4; ++role) {
                if (role == cached_ss.dof) continue;
                append_neighbor(contact.v[role]);
            }
        }

        std::sort(neighbors.begin(), neighbors.end());
    }
}

void union_adjacency(const std::vector<std::vector<int>>& a,const std::vector<std::vector<int>>& b, std::vector<std::vector<int>>& out) {
    const int nv = static_cast<int>(std::max(a.size(), b.size()));
    static const std::vector<int> empty_row;
    out.resize(nv);
    #pragma omp parallel for schedule(dynamic, 64)
    for (int vi = 0; vi < nv; ++vi) {
        const auto& row_a = vi < static_cast<int>(a.size()) ? a[vi] : empty_row;
        const auto& row_b = vi < static_cast<int>(b.size()) ? b[vi] : empty_row;
        out[vi].clear();
        out[vi].reserve(row_a.size() + row_b.size());
        std::set_union(row_a.begin(), row_a.end(), row_b.begin(), row_b.end(), std::back_inserter(out[vi]));
    }
}

void greedy_color_conflict_graph(
    const std::vector<std::vector<int>>& graph,
    std::vector<std::vector<int>>& groups,
    GreedyColoringWorkspace* reusable_workspace) {
    GreedyColoringWorkspace local_workspace;
    GreedyColoringWorkspace& workspace = reusable_workspace != nullptr ? *reusable_workspace : local_workspace;
    const int nv = static_cast<int>(graph.size());
    bool settled = false;
    // Process independent index ranges concurrently, retaining ascending
    // greedy order inside each range. Cross-range reads use the previous
    // pass. A fixed point is exactly the original ascending greedy coloring.
    // This also parallelizes the initial coloring, without requiring a cache.
    if (nv >= 128 && omp_get_max_threads() > 1) {
        if (workspace.color.size() != graph.size()) workspace.color.assign(graph.size(), -1);
        workspace.next_color.resize(graph.size());
        constexpr int range_size = 128;
        const int ranges = (nv + range_size - 1) / range_size;
        for (int round = 0; round < 4; ++round) {
            int changed = 0;
            #pragma omp parallel for schedule(dynamic, 1) reduction(|:changed)
            for (int range = 0; range < ranges; ++range) {
                const int begin = range * range_size;
                const int end = std::min(nv, begin + range_size);
                thread_local std::vector<unsigned char> used;
                for (int vi = begin; vi < end; ++vi) {
                    int color = 0;
                    if (graph[vi].size() < 64) {
                        std::uint64_t mask = 0;
                        for (const int neighbor : graph[vi]) {
                            if (neighbor < 0 || neighbor >= vi) continue;
                            const int previous = neighbor >= begin
                                ? workspace.next_color[neighbor] : workspace.color[neighbor];
                            if (previous >= 0 && previous < 64)
                                mask |= std::uint64_t(1) << previous;
                        }
                        // Fewer than 64 neighbors leave at least one free bit.
                        color = __builtin_ctzll(~mask);
                    } else {
                        used.assign(graph[vi].size() + 1, 0);
                        for (const int neighbor : graph[vi]) {
                            if (neighbor < 0 || neighbor >= vi) continue;
                            const int previous = neighbor >= begin
                                ? workspace.next_color[neighbor] : workspace.color[neighbor];
                            if (previous >= 0 && static_cast<std::size_t>(previous) < used.size())
                                used[previous] = 1;
                        }
                        while (used[color]) ++color;
                    }
                    workspace.next_color[vi] = color;
                    changed |= color != workspace.color[vi];
                }
            }
            workspace.color.swap(workspace.next_color);
            if (!changed) { settled = true; break; }
        }
    }
    if (!settled) {
        // Long dependency chains may need many parallel repair rounds. Bound
        // that overhead with the original linear greedy pass.
        workspace.color.assign(static_cast<std::size_t>(nv), -1);
        workspace.seen_color.assign(static_cast<std::size_t>(nv), -1);
        for (int vi = 0; vi < nv; ++vi) {
            for (int nb : graph[vi]) {
                if (nb >= 0 && nb < vi)
                    workspace.seen_color[workspace.color[nb]] = vi;
            }
            int color = 0;
            while (workspace.seen_color[color] == vi) ++color;
            workspace.color[vi] = color;
        }
    }
    // Small graphs spend more time in the parallel prefix passes than in
    // grouping. Append in vertex order to preserve the same sweep order.
    if (nv < 4096) {
        int max_color = -1;
        for (const int color : workspace.color) max_color = std::max(max_color, color);
        groups.resize(max_color + 1);
        for (auto& group : groups) group.clear();
        for (int vi = 0; vi < nv; ++vi) groups[workspace.color[vi]].push_back(vi);
        return;
    }
    int max_color = -1;
    #pragma omp parallel for schedule(static) reduction(max:max_color) if(nv >= 128)
    for (int vi = 0; vi < nv; ++vi) max_color = std::max(max_color, workspace.color[vi]);
    const int num_groups = max_color + 1;
    groups.resize(num_groups);
    const int workers = nv >= 128 ? omp_get_max_threads() : 1;
    std::vector<std::size_t> offsets(static_cast<std::size_t>(workers) * num_groups, 0);
    #pragma omp parallel for schedule(static, 1) if(workers > 1)
    for (int worker = 0; worker < workers; ++worker) {
        const int begin = static_cast<int>(static_cast<std::size_t>(nv) * worker / workers);
        const int end = static_cast<int>(static_cast<std::size_t>(nv) * (worker + 1) / workers);
        for (int vi = begin; vi < end; ++vi)
            ++offsets[static_cast<std::size_t>(worker) * num_groups + workspace.color[vi]];
    }
    #pragma omp parallel for schedule(static) if(nv >= 128)
    for (int color = 0; color < num_groups; ++color) {
        std::size_t offset = 0;
        for (int worker = 0; worker < workers; ++worker) {
            auto& slot = offsets[static_cast<std::size_t>(worker) * num_groups + color];
            const auto count = slot;
            slot = offset;
            offset += count;
        }
        groups[color].resize(offset);
    }
    #pragma omp parallel for schedule(static, 1) if(workers > 1)
    for (int worker = 0; worker < workers; ++worker) {
        const int begin = static_cast<int>(static_cast<std::size_t>(nv) * worker / workers);
        const int end = static_cast<int>(static_cast<std::size_t>(nv) * (worker + 1) / workers);
        for (int vi = begin; vi < end; ++vi) {
            const int color = workspace.color[vi];
            groups[color][offsets[static_cast<std::size_t>(worker) * num_groups + color]++] = vi;
        }
    }
}

namespace solver_detail {
void evaluate_contact_ranges(int count, const std::function<void(int, int)>& evaluate, int alignment, const std::function<void()>* leader_work) {
    if (active_contact_task_group != nullptr) {
        ContactTaskGroup* group = active_contact_task_group;
        active_contact_task_group = nullptr;
        struct Restore {
            ContactTaskGroup* group;
            ~Restore() { active_contact_task_group = group; }
        } restore{group};
        group->dispatch(count, evaluate, alignment, leader_work);
        return;
    }
    if (leader_work) (*leader_work)();
    std::exception_ptr error;
    int first_error = count;
    const int desired = std::max(16, count / (4 * omp_get_num_threads()));
    const int grain = (desired + alignment - 1) / alignment * alignment;
#pragma omp taskgroup
    {
        for (int begin = 0; begin < count; begin += grain) {
            const int end = std::min(begin + grain, count);
#pragma omp task shared(evaluate, error, first_error) firstprivate(begin, end)
            {
                try {
                    // A task may run on another leader. Its nested evaluation
                    // must not borrow that leader's unrelated helper group.
                    ContactTaskGroup* previous = active_contact_task_group;
                    active_contact_task_group = nullptr;
                    struct RestoreTaskGroup {
                        ContactTaskGroup* previous;
                        ~RestoreTaskGroup() { active_contact_task_group = previous; }
                    } restore{previous};
                    evaluate(begin, end);
                } catch (...) {
#pragma omp critical(ipc_contact_task_error)
                    {
                        // Each range visits contacts in order; the earliest
                        // failing range contains the earliest failing contact.
                        if (begin < first_error) {
                            first_error = begin;
                            error = std::current_exception();
                        }
                    }
                }
            }
        }
    }
    if (error) std::rethrow_exception(error);
}
} // namespace solver_detail
