#pragma once

#include "IPC_math.h"
#include "physics.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

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

struct BVHNode {
    AABB bbox;
    int left = -1;
    int right = -1;
    int leafIndex = -1;
};

int  build_bvh(const std::vector<AABB>& boxes, std::vector<BVHNode>& out);
void refit_bvh(std::vector<BVHNode>& nodes, const std::vector<AABB>& boxes);
void query_bvh(const std::vector<BVHNode>& nodes, int root, const AABB& query, std::vector<int>& hits);

// Swept-AABB broad phase producing candidate node–triangle and segment–segment pairs.
class BroadPhase {
public:
    struct Cache {
        std::vector<AABB> node_boxes;
        std::vector<AABB> tri_boxes;
        std::vector<AABB> edge_boxes;

        std::vector<BVHNode> tri_bvh_nodes;
        std::vector<BVHNode> edge_bvh_nodes;
        std::vector<BVHNode> node_bvh_nodes;

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

        struct VertexPairEntry {
            std::size_t pair_index;
            int dof;  // 0=node/v[0], 1=tri_v[0]/v[1], 2=tri_v[1]/v[2], 3=tri_v[2]/v[3]
        };
        std::vector<std::vector<VertexPairEntry>> vertex_nt;
        std::vector<std::vector<VertexPairEntry>> vertex_ss;
    };

    void initialize(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt, double dhat);

    const std::vector<NodeTrianglePair>& nt_pairs() const {
        return cache_.nt_pairs;
    }

    const std::vector<SegmentSegmentPair>& ss_pairs() const {
        return cache_.ss_pairs;
    }

    void build_ccd_candidates(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt);

    // Cache static mesh topology; reused by later build/initialize calls.
    void set_mesh_topology(const RefMesh& mesh, int nv);
    bool has_topology() const { return topology_valid_; }

    // Broad-phase candidate pairs that involve vertex vi, grouped by
    // vi's topological role. Used by both CCD and trust-region narrow
    // phases as a per-vertex pair source.
    //   nt_node_pairs : vi is the lone moving node vs an external triangle
    //   nt_face_pairs : vi is a corner of a moving triangle vs an external node
    //   ss_pairs      : vi is an endpoint of one edge vs a non-sharing edge
    struct VertexPairs {
        struct NTPair      { int node; int tri_v[3]; };
        struct NTFacePair  { int node; int tri_v[3]; int vi_local; }; // vi_local in {0,1,2}
        struct SSPair      { int v[4]; int vi_dof; };
        std::vector<NTPair>     nt_node_pairs;
        std::vector<NTFacePair> nt_face_pairs;
        std::vector<SSPair>     ss_pairs;
    };

    VertexPairs query_pairs_for_vertex(const std::vector<Vec3>& x, int vi, const Vec3& dx, const RefMesh& mesh) const;

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

    // Increments on every initialize() call — a cache-invalidation key.
    std::uint64_t version() const { return version_; }

private:
    Cache cache_;
    bool topology_valid_ = false;
    std::uint64_t version_ = 0;

    // Static mesh connectivity, reused across every build for the same mesh.
    struct Topology {
        std::vector<std::array<int, 2>> edges;
        std::vector<std::vector<int>> node_to_edges;
        std::vector<std::vector<int>> node_to_tris;
    };
    Topology topo_;

    void build(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt, double node_pad, double tri_pad, double edge_pad);
};
