#include "bvh_box_leaf.h"
#include <algorithm>
#include <limits>

// =============================================================================
// AABB
// =============================================================================

// Default-construct to an "inside-out" box: min = +inf, max = -inf.
// This is the identity element for expand() — any real point or box will
// immediately override both corners on the first expand call.
AABB::AABB() {
    double inf = std::numeric_limits<double>::infinity();
    min = Vec2(inf, inf);
    max = Vec2(-inf, -inf);
}

AABB::AABB(const Vec2& a, const Vec2& b) : min(a), max(b) {}

// Grow this box to also contain box o.
void AABB::expand(const AABB& o) {
    min.x = std::min(min.x, o.min.x); min.y = std::min(min.y, o.min.y);
    max.x = std::max(max.x, o.max.x); max.y = std::max(max.y, o.max.y);
}

// Grow this box to also contain point p.
void AABB::expand(const Vec2& p) {
    min.x = std::min(min.x, p.x); min.y = std::min(min.y, p.y);
    max.x = std::max(max.x, p.x); max.y = std::max(max.y, p.y);
}

Vec2 AABB::centroid() const { return (min + max) * 0.5; }
Vec2 AABB::extent()   const { return max - min; }

// =============================================================================
// BVH build
//
// The tree is stored as a flat array of BVHNode. Every recursive call
// claims the next slot in the array before recursing, so a parent's index
// is always lower than its children's indices. This property is exploited
// by refit_bvh() to do a bottom-up pass without storing parent pointers.
//
// Build strategy: median split on the longest axis.
//   1. Compute the bounding box of all boxes in [start, end).
//   2. If only one box remains, make a leaf.
//   3. Otherwise choose the axis (X or Y) along which the combined box is
//      widest, then partition the index array at the median centroid using
//      std::nth_element (O(n) average). Recurse on each half.
//
// Time:  O(n log n) average (nth_element at each level of a balanced tree)
// Space: O(n) nodes  (2n - 1 for n leaves)
// =============================================================================

static int build_recursive(std::vector<BVHNode>& nodes,
                            const std::vector<AABB>& boxes,
                            std::vector<int>& indices, int start, int end) {
    // Claim a node slot before any recursion so this node's index is lower
    // than any node created during the recursive calls below.
    int nodeIdx = static_cast<int>(nodes.size());
    nodes.emplace_back();

    // Compute the tight bounding box over all boxes in this range.
    AABB nodeBox;
    for (int i = start; i < end; ++i) nodeBox.expand(boxes[indices[i]]);
    nodes[nodeIdx].bbox = nodeBox;

    int count = end - start;

    // Base case: single box → leaf node.
    if (count == 1) {
        nodes[nodeIdx].leafIndex = indices[start];
        return nodeIdx;
    }

    // Choose the axis with the greatest extent so the split is as even as
    // possible and the resulting child boxes overlap as little as possible.
    Vec2 e = nodeBox.extent();
    int axis = (e.y > e.x) ? 1 : 0;   // 0 = split along X, 1 = split along Y

    // Partition indices so that the first half has the boxes whose centroid
    // lies below the median on the chosen axis, and the second half above.
    // nth_element guarantees the element at 'mid' is in its sorted position
    // and everything before it is ≤ it — O(n) average, no full sort needed.
    int mid = start + count / 2;
    auto cmp = [&](int a, int b) {
        Vec2 ca = boxes[a].centroid(), cb = boxes[b].centroid();
        return axis == 0 ? ca.x < cb.x : ca.y < cb.y;
    };
    std::nth_element(indices.begin()+start, indices.begin()+mid, indices.begin()+end, cmp);

    // Recurse. left/right store the indices of the child nodes in the flat array.
    int left  = build_recursive(nodes, boxes, indices, start, mid);
    int right = build_recursive(nodes, boxes, indices, mid,   end);
    nodes[nodeIdx].left  = left;
    nodes[nodeIdx].right = right;
    return nodeIdx;
}

// Public entry point. Builds a BVH over the given boxes and returns the root
// node index (always 0 for a non-empty input).
int build_bvh(const std::vector<AABB>& boxes, std::vector<BVHNode>& out) {
    out.clear();
    if (boxes.empty()) return -1;
    // Work on an index array so the original boxes vector is never reordered.
    std::vector<int> idx(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) idx[i] = static_cast<int>(i);
    return build_recursive(out, boxes, idx, 0, static_cast<int>(idx.size()));
}

// =============================================================================
// Refit
//
// After the leaf AABBs change (e.g. objects moved), propagate the new sizes
// up through the tree without rebuilding the topology.
//
// Because build_recursive assigns parent indices before child indices, parents
// always have lower indices than their children. Iterating from the end of the
// array back to 0 therefore processes every child before its parent — a
// bottom-up pass with no extra bookkeeping.
// =============================================================================

void refit_bvh(std::vector<BVHNode>& nodes, const std::vector<AABB>& boxes) {
    for (int i = static_cast<int>(nodes.size())-1; i >= 0; --i) {
        BVHNode& n = nodes[i];
        if (n.leafIndex >= 0) {
            // Leaf: pull the updated AABB directly from the source data.
            n.bbox = boxes[n.leafIndex];
        } else {
            // Internal: recompute as the union of the (already updated) children.
            n.bbox = AABB{};
            n.bbox.expand(nodes[n.left].bbox);
            n.bbox.expand(nodes[n.right].bbox);
        }
    }
}

// =============================================================================
// Query
// =============================================================================

// Returns true if two AABBs overlap (touching edges count as overlap).
// Separation on any single axis is sufficient to prove no intersection,
// so we test both axes and return false as soon as a gap is found.
bool aabb_intersects(const AABB& a, const AABB& b) {
    return a.min.x <= b.max.x && a.max.x >= b.min.x &&
           a.min.y <= b.max.y && a.max.y >= b.min.y;
}

// Recursive BVH traversal. Appends to 'hits' the index of every input box
// whose leaf AABB overlaps the query box.
//
// Early-out: if the current node's bbox doesn't intersect the query we skip
// the entire subtree — this is where the O(log n + k) speedup over brute
// force comes from (k = number of hits).
void query_bvh(const std::vector<BVHNode>& nodes, int nodeIdx,
               const AABB& query, std::vector<int>& hits) {
    if (nodeIdx < 0) return;
    const BVHNode& n = nodes[nodeIdx];

    // Prune: this node's bbox doesn't touch the query, so nothing beneath it
    // can either — skip the whole subtree.
    if (!aabb_intersects(n.bbox, query)) return;

    if (n.leafIndex >= 0) {
        // Leaf that passed the intersection test — record it.
        hits.push_back(n.leafIndex);
        return;
    }

    // Internal node: descend into both children.
    query_bvh(nodes, n.left,  query, hits);
    query_bvh(nodes, n.right, query, hits);
}
