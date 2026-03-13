#include "bvh.h"
#include <algorithm>

using Cache = BVHBroadPhase::Cache;
using NSP   = physics::NodeSegmentPair;

// ======================================================
// BVH tree — build, refit, query
// ======================================================

static int build_recursive(std::vector<BVHNode>& nodes,
                            const std::vector<AABB>& boxes,
                            std::vector<int>& indices, int start, int end) {
    int nodeIdx = static_cast<int>(nodes.size());
    nodes.emplace_back();

    AABB nodeBox;
    for (int i = start; i < end; ++i) nodeBox.expand(boxes[indices[i]]);
    nodes[nodeIdx].bbox = nodeBox;

    int count = end - start;
    if (count == 1) {
        nodes[nodeIdx].leafIndex = indices[start];
        return nodeIdx;
    }

    Vec2 e = nodeBox.extent();
    int axis = (e.y > e.x) ? 1 : 0;

    int mid = start + count / 2;
    auto cmp = [&](int a, int b) {
        Vec2 ca = boxes[a].centroid(), cb = boxes[b].centroid();
        return axis == 0 ? ca.x < cb.x : ca.y < cb.y;
    };
    std::nth_element(indices.begin()+start, indices.begin()+mid, indices.begin()+end, cmp);

    int left  = build_recursive(nodes, boxes, indices, start, mid);
    int right = build_recursive(nodes, boxes, indices, mid,   end);
    nodes[nodeIdx].left  = left;
    nodes[nodeIdx].right = right;
    return nodeIdx;
}

int build_bvh(const std::vector<AABB>& boxes, std::vector<BVHNode>& out) {
    out.clear();
    if (boxes.empty()) return -1;
    std::vector<int> idx(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) idx[i] = static_cast<int>(i);
    return build_recursive(out, boxes, idx, 0, static_cast<int>(idx.size()));
}

void refit_bvh(std::vector<BVHNode>& nodes, const std::vector<AABB>& boxes) {
    for (int i = static_cast<int>(nodes.size())-1; i >= 0; --i) {
        BVHNode& n = nodes[i];
        if (n.leafIndex >= 0) {
            n.bbox = boxes[n.leafIndex];
        } else {
            n.bbox = AABB{};
            n.bbox.expand(nodes[n.left].bbox);
            n.bbox.expand(nodes[n.right].bbox);
        }
    }
}

void query_bvh(const std::vector<BVHNode>& nodes, int nodeIdx,
               const AABB& query, std::vector<int>& hits) {
    if (nodeIdx < 0) return;
    const BVHNode& n = nodes[nodeIdx];
    if (!aabb_intersects(n.bbox, query)) return;
    if (n.leafIndex >= 0) {
        hits.push_back(n.leafIndex);
        return;
    }
    query_bvh(nodes, n.left,  query, hits);
    query_bvh(nodes, n.right, query, hits);
}

// ======================================================
// Internal helpers (file-local, operate on Cache)
// ======================================================

static inline bool is_valid_segment_start(int j, int N_left, int N_right) {
    const int total = N_left + N_right;
    if (j < 0 || j >= total - 1) return false;
    return (j < N_left - 1) || (j >= N_left && j < N_left + N_right - 1);
}

static inline bool is_invalid_pair(int node, int seg0, int seg1) {
    return (node == seg0) || (node == seg1);
}

static inline std::uint64_t pair_key(int node, int seg0) {
    return (std::uint64_t(std::uint32_t(node)) << 32) | std::uint32_t(seg0);
}

static inline AABB build_node_box(const Vec& x, const Vec& v, int i, double dt, double pad) {
    Vec2 x0 = get_xi(x, i), vi = get_xi(v, i);
    Vec2 x1{x0.x + dt * vi.x, x0.y + dt * vi.y};
    return AABB(Vec2(std::min(x0.x, x1.x) - pad, std::min(x0.y, x1.y) - pad),
                Vec2(std::max(x0.x, x1.x) + pad, std::max(x0.y, x1.y) + pad));
}

static inline AABB build_segment_box(const Vec& x, const Vec& v, int seg0, double dt, double pad) {
    Vec2 x0 = get_xi(x, seg0), x1 = get_xi(x, seg0 + 1);
    Vec2 v0 = get_xi(v, seg0), v1 = get_xi(v, seg0 + 1);
    return AABB(Vec2(std::min({x0.x, x1.x, x0.x + dt*v0.x, x1.x + dt*v1.x}) - pad,
                     std::min({x0.y, x1.y, x0.y + dt*v0.y, x1.y + dt*v1.y}) - pad),
                Vec2(std::max({x0.x, x1.x, x0.x + dt*v0.x, x1.x + dt*v1.x}) + pad,
                     std::max({x0.y, x1.y, x0.y + dt*v0.y, x1.y + dt*v1.y}) + pad));
}

static inline void add_pair(Cache& c, int node, int seg0) {
    std::uint64_t key = pair_key(node, seg0);
    if (c.pair_index.count(key)) return;
    c.pair_index[key] = c.pairs.size();
    c.pairs.push_back({node, seg0, seg0 + 1});
}

static inline void erase_pair_at(Cache& c, std::size_t idx) {
    const std::size_t last = c.pairs.size() - 1;
    NSP victim = c.pairs[idx];
    if (idx != last) {
        c.pairs[idx] = c.pairs[last];
        c.pair_index[pair_key(c.pairs[idx].node, c.pairs[idx].seg0)] = idx;
    }
    c.pairs.pop_back();
    c.pair_index.erase(pair_key(victim.node, victim.seg0));
}

static inline void remove_pairs_touching_node(Cache& c, int node) {
    for (std::size_t i = c.pairs.size(); i > 0; --i)
        if (c.pairs[i-1].node == node) erase_pair_at(c, i-1);
}

static inline void remove_pairs_touching_segment(Cache& c, int seg0) {
    for (std::size_t i = c.pairs.size(); i > 0; --i)
        if (c.pairs[i-1].seg0 == seg0) erase_pair_at(c, i-1);
}

static inline void query_node(Cache& c, int node, int N_left, int N_right) {
    if (c.seg_bvh_root < 0) return;
    const int total = N_left + N_right;
    std::vector<int> hits;
    query_bvh(c.seg_bvh_nodes, c.seg_bvh_root, c.node_boxes[node], hits);
    for (int leaf_k : hits) {
        int seg0 = c.seg_leaf_to_seg0[leaf_k], seg1 = seg0 + 1;
        if (seg1 >= total || is_invalid_pair(node, seg0, seg1)) continue;
        add_pair(c, node, seg0);
    }
}

static inline void scan_segment(Cache& c, int seg0, int N_left, int N_right) {
    if (!is_valid_segment_start(seg0, N_left, N_right)) return;
    const int total = N_left + N_right, seg1 = seg0 + 1;
    const AABB& sb = c.segment_boxes[seg0];
    for (int node = 0; node < total; ++node)
        if (!is_invalid_pair(node, seg0, seg1) && aabb_intersects(c.node_boxes[node], sb))
            add_pair(c, node, seg0);
}

// ======================================================
// BVHBroadPhase implementation
// ======================================================

void BVHBroadPhase::build(const Vec& x, const Vec& v,
                          int N_left, int N_right, double dt,
                          double node_pad, double seg_pad) {
    N_left_  = N_left;
    N_right_ = N_right;

    Cache c;
    const int total = N_left + N_right, nseg = std::max(0, total - 1);

    c.node_boxes.resize(total);
    c.segment_boxes.resize(nseg);
    c.segment_valid.assign(nseg, 0);
    c.seg0_to_leaf.assign(nseg, -1);

    for (int i = 0; i < total; ++i)
        c.node_boxes[i] = build_node_box(x, v, i, dt, node_pad);

    for (int j = 0; j < nseg; ++j) {
        if (!is_valid_segment_start(j, N_left, N_right)) continue;
        c.segment_boxes[j] = build_segment_box(x, v, j, dt, seg_pad);
        c.segment_valid[j] = 1;
        int leaf_k = static_cast<int>(c.seg_leaf_to_seg0.size());
        c.seg0_to_leaf[j] = leaf_k;
        c.seg_leaf_to_seg0.push_back(j);
        c.seg_bvh_boxes.push_back(c.segment_boxes[j]);
    }

    c.seg_bvh_root = build_bvh(c.seg_bvh_boxes, c.seg_bvh_nodes);

    for (int i = 0; i < total; ++i)
        query_node(c, i, N_left, N_right);

    cache_ = std::move(c);
}

void BVHBroadPhase::initialize(const Vec& x, const Vec& v,
                               int N_left, int N_right, double dt, double dhat) {
    build(x, v, N_left, N_right, dt, /*node_pad=*/dhat, /*seg_pad=*/0.0);
}

void BVHBroadPhase::refresh(const Vec& x, const Vec& v,
                            int moved_node, int N_left, int N_right,
                            double dt, double node_pad, double seg_pad) {
    const int total = N_left + N_right;
    if (moved_node < 0 || moved_node >= total) return;

    cache_.node_boxes[moved_node] = build_node_box(x, v, moved_node, dt, node_pad);

    const int left_seg = moved_node - 1, right_seg = moved_node;

    auto update_seg = [&](int seg0) {
        if (!is_valid_segment_start(seg0, N_left, N_right)) return;
        cache_.segment_boxes[seg0] = build_segment_box(x, v, seg0, dt, seg_pad);
        int leaf_k = cache_.seg0_to_leaf[seg0];
        if (leaf_k >= 0) cache_.seg_bvh_boxes[leaf_k] = cache_.segment_boxes[seg0];
    };

    update_seg(left_seg);
    update_seg(right_seg);

    if (cache_.seg_bvh_root >= 0)
        refit_bvh(cache_.seg_bvh_nodes, cache_.seg_bvh_boxes);

    remove_pairs_touching_node(cache_, moved_node);
    if (is_valid_segment_start(left_seg,  N_left, N_right)) remove_pairs_touching_segment(cache_, left_seg);
    if (is_valid_segment_start(right_seg, N_left, N_right)) remove_pairs_touching_segment(cache_, right_seg);

    query_node(cache_, moved_node, N_left, N_right);
    scan_segment(cache_, left_seg,  N_left, N_right);
    scan_segment(cache_, right_seg, N_left, N_right);
}

const std::vector<NSP>& BVHBroadPhase::pairs() const {
    return cache_.pairs;
}

std::vector<NSP> BVHBroadPhase::build_ccd_candidates(const Vec& x, const Vec& v,
                                                      int N_left, int N_right, double dt) {
    BVHBroadPhase tmp;
    tmp.build(x, v, N_left, N_right, dt, /*node_pad=*/0.0, /*seg_pad=*/0.0);
    return tmp.cache_.pairs;
}

std::vector<NSP> BVHBroadPhase::build_trust_region_candidates(const Vec& x, const Vec& v,
                                                               int N_left, int N_right,
                                                               double dt, double motion_pad) {
    BVHBroadPhase tmp;
    tmp.build(x, v, N_left, N_right, dt, /*node_pad=*/motion_pad, /*seg_pad=*/0.0);
    return tmp.cache_.pairs;
}
