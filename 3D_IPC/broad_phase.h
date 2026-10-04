#pragma once

#include "IPC_math.h"
#include "physics.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
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

// Conservative AABB and supporting-plane/line rejection before exact primitive-distance evaluation.
// Optional output distinguishes an AABB-distance rejection from the later
// plane-distance rejection. Only the former certifies separated axis bounds.
bool node_triangle_aabbs_within_distance(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c, double distance_squared, bool* aabb_rejected = nullptr);
bool segment_aabbs_within_distance(const Vec3& a0, const Vec3& a1, const Vec3& b0, const Vec3& b1, double distance_squared, bool* aabb_rejected = nullptr);

struct BVHNode {
    AABB bbox;
    int left = -1;
    int right = -1;
    int parent = -1;
    int leafIndex = -1;
};

int  build_bvh(const std::vector<AABB>& boxes, std::vector<BVHNode>& out);

// Same, plus fills leaf_to_node[leafIndex] = node_idx (required by refit_bvh_leaf).
int  build_bvh(const std::vector<AABB>& boxes, std::vector<BVHNode>& out, std::vector<int>& leaf_to_node);

void refit_bvh(std::vector<BVHNode>& nodes, const std::vector<AABB>& boxes);

// Refits one leaf and walks its parent chain until a parent's bbox is unchanged.
void refit_bvh_leaf(std::vector<BVHNode>& nodes, const std::vector<int>& leaf_to_node, int leafIndex, const AABB& new_box);

void query_bvh(const std::vector<BVHNode>& nodes, int root, const AABB& query, std::vector<int>& hits);

// Independent AoS query records are transposed only inside this fixed tile.
// Each output row retains the scalar traversal's exact hit/append order.
inline constexpr std::size_t bvh_query_tile_width = 8;
void query_bvh_tile(const std::vector<BVHNode>& nodes, int root, const AABB* queries, std::size_t count, std::vector<int>* const* hits);

// Swept-AABB broad phase producing candidate node–triangle and segment–segment pairs.
class BroadPhase {
public:
    enum class InitializationMode {
        // Legacy/default storage: retain every candidate, all per-vertex
        // incidence, and the node/leaf maps required by incremental refits.
        Refittable,
        // Deformable solve: retain every candidate and all per-vertex
        // incidence, but omit node/leaf storage that this solver never refits.
        DeformableSolver,
        // Mixed deformable/rigid solve: rigid self-pairs are impossible and
        // rigid proxy vertices do not need per-vertex incidence. Deformable
        // incidence is retained for local Newton and CCD updates.
        GeneralSolver,
        // Rigid-only solve: retain only cross-body candidate arrays. Per-body
        // ownership is built by the solver, so no per-vertex incidence or
        // refit-only BVH storage is needed.
        RigidSolver,
    };

    struct Cache {
        bool excludes_tet_interior_nt_queries = false;
        std::vector<AABB> node_boxes;
        std::vector<AABB> tri_boxes;
        std::vector<AABB> edge_boxes;

        std::vector<BVHNode> tri_bvh_nodes;
        std::vector<BVHNode> edge_bvh_nodes;
        std::vector<BVHNode> node_bvh_nodes;

        // leafIndex -> BVH node index
        std::vector<int> tri_leaf_to_node;
        std::vector<int> edge_leaf_to_node;
        std::vector<int> node_leaf_to_node;

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

        struct VertexPairEntry {
            // Keep each incidence in one machine word. Three signed role bits
            // leave 61 index bits on 64-bit hosts, more than any allocated pair
            // vector can contain (each pair occupies at least eight bytes).
            // This retains wide indices without a separate narrow/wide cache.
            static constexpr int pair_index_bits =
                std::numeric_limits<std::size_t>::digits - 3;
            static constexpr std::size_t max_pair_index =
                std::numeric_limits<std::size_t>::max() >> 3;
            // Index of the contact in nt_pairs or ss_pairs.
            std::size_t pair_index : pair_index_bits;
            // Role of this vertex within that four-vertex contact:
            // 0=node/v[0], 1=tri_v[0]/v[1], 2=tri_v[1]/v[2], 3=tri_v[2]/v[3].
            int dof : 3;
        };
        static_assert(sizeof(NodeTrianglePair) >= 8 && sizeof(SegmentSegmentPair) >= 8,
            "Packed incidence indices must cover every allocatable contact pair");
        static_assert(sizeof(VertexPairEntry) == sizeof(std::size_t),
            "Contact incidence should occupy one machine word");
        // Per-vertex references to the actual contact-pair arrays above.
        std::vector<std::vector<VertexPairEntry>> vertex_nt;
        std::vector<std::vector<VertexPairEntry>> vertex_ss;

        // Scratch buffers reused across broad-phase rebuilds.
        std::vector<std::vector<int>> node_hits;
        std::vector<std::vector<int>> edge_hits;
        std::vector<AABB> red_edge_boxes;
        // Per-BVH-node rigid ownership used to prune solver-only self-contact
        // queries. A nonnegative value means every leaf below the node belongs
        // to that rigid body; -1 means mixed or deformable ownership.
        std::vector<int> tri_bvh_rigid_owner;
        std::vector<int> edge_bvh_rigid_owner;
    };

    void initialize(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt, double dhat);

    // Velocity-based initialization for scenes containing tetrahedral solids.
    // NT queries skip only classified tet-interior nodes (tet_nodes minus
    // surface_nodes). Non-solid point-only proxies remain eligible, while
    // triangle and edge topology, including SS candidates, is unchanged.
    void initialize_surface_nodes(
        const std::vector<Vec3>& x, const std::vector<Vec3>& v,
        const RefMesh& mesh, double dt, double dhat);

    // Pre-built-box counterpart used by the mixed cloth/solid/rigid solver.
    // Tet-interior nodes keep their boxes for elastic updates but do not issue
    // node-triangle contact queries.
    void initialize_surface_nodes(const std::vector<AABB>& vertex_boxes, const RefMesh& mesh, double d_hat = 0.0, InitializationMode mode = InitializationMode::Refittable);

    // Initialize from pre-built per-vertex AABBs. Triangle and edge boxes are
    // derived as the union of their vertex boxes (i.e. red boxes).
    void initialize(const std::vector<AABB>& vertex_boxes, const RefMesh& mesh, double d_hat = 0.0, InitializationMode mode = InitializationMode::Refittable);

    // Install only the per-vertex motion boxes needed by node-box step
    // clipping. Contact pairs, contact-incidence arrays, primitive boxes, and
    // BVHs are cleared; reusable mesh-topology arrays remain. This performs no
    // contact search and must not be followed by incremental BVH refit/query.
    void initialize_node_boxes_only(const std::vector<AABB>& vertex_boxes);

    // Update all boxes and refit the existing BVHs without changing their
    // topology or pair lists. Requires box-based Refittable initialization
    // with the same mesh and vertex count. Call refresh_pairs afterwards.
    void refit_boxes(const std::vector<AABB>& vertex_boxes, const RefMesh& mesh, double box_pad);

    // Re-query NT and SS pair lists from the current BVH state without
    // rebuilding the BVHs. After box refits, rebuild vertex_nt/vertex_ss and
    // the ordered pair arrays for the next solver iteration.
    void refresh_pairs(const RefMesh& mesh);

    const std::vector<NodeTrianglePair>& nt_pairs() const {
        return cache_.nt_pairs;
    }

    const std::vector<SegmentSegmentPair>& ss_pairs() const {
        return cache_.ss_pairs;
    }

    // Initial guesses consume only candidate pairs; they can omit solver
    // incidence and the node BVH while retaining the same ordered pair lists.
    // The default retains the existing cache interface for other callers.
    void build_ccd_candidates(const std::vector<Vec3>& x, const std::vector<Vec3>& v, const RefMesh& mesh, double dt, bool retain_solver_data = true);

    // Cache static mesh topology; reused by later build/initialize calls.
    void set_mesh_topology(const RefMesh& mesh, int nv);
    bool has_topology() const { return topology_valid_; }

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

    // Remove incidence for pairs certified separated for every
    // position inside the current node boxes. Pair arrays/indices and BVHs stay
    // unchanged. Call after coloring to preserve the GS update order, and
    // rebuild before any vertex can leave these boxes.
    void discard_separated_contact_incidence(
        const std::vector<unsigned char>& nt_separated,
        const std::vector<unsigned char>& ss_separated,
        bool parallel,
        std::vector<std::vector<int>>* body_nt = nullptr,
        std::vector<std::vector<int>>* body_ss = nullptr);

    // Used by global_gauss_seidel_solver_ogc for partial leaf refit; other callers should use cache().
    Cache& mutable_cache() { return cache_; }

private:
    Cache cache_;
    bool topology_valid_ = false;
    // Candidate-domain policy established by the most recent initialization;
    // refresh_pairs() must rebuild pairs in the same domain.
    bool exclude_tet_interior_nt_queries_ = false;

    // Static mesh connectivity, reused across every build for the same mesh.
    struct Topology {
        std::vector<std::array<int, 2>> edges;
        std::vector<std::vector<int>> node_to_edges;
        std::vector<std::vector<int>> node_to_tris;
        std::vector<int> tri_rigid_owner;
        std::vector<int> edge_rigid_owner;
        // Ascending list of every node allowed to issue an NT query under the
        // surface-node policy. This includes boundary nodes and non-solid
        // point proxies, and excludes only tet-interior nodes.
        std::vector<int> surface_nt_query_nodes;
        bool surface_nt_query_nodes_valid = false;
    };
    Topology topo_;

    void initialize_from_vertex_boxes(const std::vector<AABB>& vertex_boxes, const RefMesh& mesh, double d_hat, bool exclude_tet_interior_nt_queries, InitializationMode mode);
    const std::vector<int>& surface_nt_query_nodes(
        const RefMesh& mesh, int nv);

    void build(
        const std::vector<Vec3>& x, const std::vector<Vec3>& v,
        const RefMesh& mesh, double dt, double node_pad, double tri_pad,
        double edge_pad, bool exclude_tet_interior_nt_queries,
        bool retain_solver_data = true);
};

void incremental_refresh_vertex(BroadPhase::Cache& c, int vi, const std::vector<Vec3>& x, const RefMesh& mesh, double box_pad, double node_box_radius_padded);
