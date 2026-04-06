#pragma once

#include "IPC_math.h"
#include "physics.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

// Axis-aligned bounding box
struct AABB {
    Vec3 min, max;

    AABB() {
        const double inf = std::numeric_limits<double>::infinity();
        min = Vec3( inf,  inf,  inf);
        max = Vec3(-inf, -inf, -inf);
    }

    AABB(const Vec3& a, const Vec3& b) : min(a), max(b) {}

    void expand(const Vec3& p) {
        min = min.cwiseMin(p);
        max = max.cwiseMax(p);
    }

    void expand(const AABB& box) {
        expand(box.min);
        expand(box.max);
    }

    Vec3 centroid() const {
        return 0.5 * (min + max);
    }

    Vec3 extent() const {
        return max - min;
    }
};

inline bool aabb_intersects(const AABB& a, const AABB& b) {
    return (a.min.array() <= b.max.array()).all() && (a.max.array() >= b.min.array()).all();
}

// Flat-array BVH node
struct BVHNode {
    AABB bbox;
    int left = -1;
    int right = -1;
    int leafIndex = -1;
};

int  build_bvh(const std::vector<AABB>& boxes, std::vector<BVHNode>& out);
void refit_bvh(std::vector<BVHNode>& nodes, const std::vector<AABB>& boxes);
void query_bvh(const std::vector<BVHNode>& nodes, int root, const AABB& query, std::vector<int>& hits);

// Broad phase: builds candidate node-triangle and segment-segment pairs from swept AABBs over a motion interval
class BroadPhase {
public:
    struct Cache {
        std::vector<AABB> node_boxes;
        std::vector<AABB> tri_boxes;
        std::vector<AABB> edge_boxes;

        std::vector<BVHNode> tri_bvh_nodes;
        std::vector<BVHNode> edge_bvh_nodes;
        std::vector<BVHNode> node_bvh_nodes;
        std::vector<int> node_bvh_parent;
        std::vector<int> node_leaf_node;
        std::vector<int> tri_bvh_parent;
        std::vector<int> edge_bvh_parent;
        std::vector<int> tri_leaf_node;
        std::vector<int> edge_leaf_node;

        int node_root = -1;
        int tri_root = -1;
        int edge_root = -1;

        std::vector<std::array<int, 2>> edges;
        std::vector<std::vector<int>> node_to_tris;
        std::vector<std::vector<int>> node_to_edges;

        std::vector<NodeTrianglePair> nt_pairs;
        std::vector<SegmentSegmentPair> ss_pairs;

        std::vector<int> nt_pair_tri;
        std::vector<std::array<int, 2>> ss_pair_edges;

        std::unordered_map<std::uint64_t, std::size_t> nt_pair_index;
        std::unordered_map<std::uint64_t, std::size_t> ss_pair_index;
    };

    void initialize(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt, double dhat);

    void refresh(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, int moved_node, double dt,
                 double node_pad, double tri_pad, double edge_pad);

    const std::vector<NodeTrianglePair>& nt_pairs() const {
        return cache_.nt_pairs;
    }

    const std::vector<SegmentSegmentPair>& ss_pairs() const {
        return cache_.ss_pairs;
    }

    void build_ccd_candidates(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt,
                              std::vector<NodeTrianglePair>& out_nt, std::vector<SegmentSegmentPair>& out_ss);

    static std::uint64_t nt_key(int node, int tri) {
        return (std::uint64_t(std::uint32_t(node)) << 32) |
               std::uint32_t(tri);
    }

    static std::uint64_t ss_key(int e0, int e1) {
        if (e0 > e1) std::swap(e0, e1);
        return (std::uint64_t(std::uint32_t(e0)) << 32) |
               std::uint32_t(e1);
    }

    const Cache& cache() const {
        return cache_;
    }

private:
    Cache cache_;

    void build(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt, double node_pad, double tri_pad, double edge_pad);
};
