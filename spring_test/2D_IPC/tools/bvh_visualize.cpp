// Visualizes a BVH as an SVG file.
// Compile: clang++ -std=c++17 -O2 bvh_visualize.cpp bvh_box_leaf.cpp -o bvh_viz && ./bvh_viz
// Then open bvh_out.svg in a browser.

#include "bvh_box_leaf.h"
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

static const char* DEPTH_COLOR[] = {
    "#ff3333", // depth 0 - root
    "#00e5ff", // depth 1
    "#ffea00", // depth 2
    "#00ff88", // depth 3
    "#ff66ff", // depth 4
    "#ff9900", // depth 5
};
static const int NUM_COLORS = 6;
static const char* LEAF_COLOR  = "#aaffaa";
static const char* GRID_COLOR  = "#2a2a4a";
static const char* AXIS_COLOR  = "#666688";

// ---- SVG output ------------------------------------------------------------

static std::ofstream SVG;
static double SCALE, PAD, WMIN_X, WMIN_Y, WMAX_Y;

static double sx(double x) { return (x - WMIN_X) * SCALE + PAD; }
static double sy(double y) { return (WMAX_Y - y) * SCALE + PAD; }   // Y-flip

static void svg_rect(const AABB& b, const char* color,
                     double fill_op, double stroke_op, double sw) {
    double x = sx(b.min.x), y = sy(b.max.y);
    double w = (b.max.x - b.min.x) * SCALE;
    double h = (b.max.y - b.min.y) * SCALE;
    SVG << "  <rect x='" << x << "' y='" << y
        << "' width='" << w << "' height='" << h << "'"
        << " fill='" << color << "' fill-opacity='" << fill_op << "'"
        << " stroke='" << color << "' stroke-opacity='" << stroke_op << "'"
        << " stroke-width='" << sw << "'/>\n";
}

static void svg_text(double x, double y, const char* color, int fs,
                     const char* anchor, const std::string& s) {
    SVG << "  <text x='" << x << "' y='" << y << "'"
        << " fill='" << color << "'"
        << " font-family='monospace' font-size='" << fs << "'"
        << " text-anchor='" << anchor << "'>" << s << "</text>\n";
}

// ---- Grid + axis labels ----------------------------------------------------

static void draw_grid(double wmin_x, double wmax_x, double wmin_y, double wmax_y) {
    // vertical lines
    for (double x = wmin_x; x <= wmax_x; x += 1.0) {
        double sx_ = sx(x);
        double top = sy(wmax_y), bot = sy(wmin_y);
        SVG << "  <line x1='" << sx_ << "' y1='" << top
            << "' x2='" << sx_ << "' y2='" << bot
            << "' stroke='" << GRID_COLOR << "' stroke-width='1'/>\n";
        svg_text(sx_, bot + 14, AXIS_COLOR, 10, "middle", std::to_string((int)x));
    }
    // horizontal lines
    for (double y = wmin_y; y <= wmax_y; y += 1.0) {
        double sy_ = sy(y);
        double left = sx(wmin_x), right = sx(wmax_x);
        SVG << "  <line x1='" << left << "' y1='" << sy_
            << "' x2='" << right << "' y2='" << sy_
            << "' stroke='" << GRID_COLOR << "' stroke-width='1'/>\n";
        svg_text(left - 6, sy_ + 4, AXIS_COLOR, 10, "end", std::to_string((int)y));
    }
}

// ---- Legend ----------------------------------------------------------------

static void draw_legend(int max_internal_depth, double legend_x, double svg_h) {
    double lx = legend_x + 16, ly = PAD;
    double row = 28, sw = 16;
    int rows = max_internal_depth + 1 + 1; // depths + leaf

    SVG << "  <rect x='" << lx - 14 << "' y='" << ly - 10
        << "' width='190' height='" << rows * row + 32 << "'"
        << " rx='6' fill='#1e1e35' stroke='#555' stroke-width='1'/>\n";
    svg_text(lx + 78, ly + 8, "#cccccc", 13, "middle", "Legend");
    ly += row;

    for (int d = 0; d <= max_internal_depth; ++d) {
        const char* c = DEPTH_COLOR[d % NUM_COLORS];
        SVG << "  <rect x='" << lx << "' y='" << ly
            << "' width='" << sw << "' height='" << sw << "'"
            << " fill='none' stroke='" << c << "' stroke-width='3'/>\n";
        std::string label = "Depth " + std::to_string(d) +
                            (d == 0 ? " (root)" : "");
        svg_text(lx + sw + 8, ly + 13, c, 12, "start", label);
        ly += row;
    }
    // leaf swatch
    SVG << "  <rect x='" << lx << "' y='" << ly
        << "' width='" << sw << "' height='" << sw << "'"
        << " fill='" << LEAF_COLOR << "' fill-opacity='0.35'"
        << " stroke='" << LEAF_COLOR << "' stroke-width='2'/>\n";
    svg_text(lx + sw + 8, ly + 13, LEAF_COLOR, 12, "start", "Leaf");
}

// ---- BVH drawing -----------------------------------------------------------

// Collect nodes level by level (BFS) so we can draw outermost depth first
// without children overwriting parent strokes.
static void draw_bvh(const std::vector<BVHNode>& nodes, int root) {
    // BFS: draw each depth level completely before moving to the next
    std::queue<std::pair<int,int>> q; // {nodeIdx, depth}
    q.push({root, 0});
    while (!q.empty()) {
        auto [idx, depth] = q.front(); q.pop();
        const BVHNode& n = nodes[idx];
        if (n.leafIndex >= 0) continue; // leaves drawn separately

        const char* color = DEPTH_COLOR[depth % NUM_COLORS];
        double sw = std::max(1.5, 5.0 - depth * 1.2);
        // stroke-only — no fill — so we never paint over adjacent boxes
        svg_rect(n.bbox, color, 0.0, 1.0, sw);

        if (n.left  >= 0) q.push({n.left,  depth + 1});
        if (n.right >= 0) q.push({n.right, depth + 1});
    }
}

static void draw_leaves(const std::vector<BVHNode>& nodes) {
    for (const BVHNode& n : nodes) {
        if (n.leafIndex < 0) continue;
        const AABB& b = n.bbox;
        svg_rect(b, LEAF_COLOR, 0.25, 1.0, 2.0);

        // Label at center, clamped so it stays inside the box
        double cx = sx((b.min.x + b.max.x) * 0.5);
        double box_top_svg    = sy(b.max.y);
        double box_bottom_svg = sy(b.min.y);
        double box_h_svg      = box_bottom_svg - box_top_svg;
        double cy = box_top_svg + box_h_svg * 0.5 + 5; // center + half font cap
        svg_text(cx, cy, LEAF_COLOR, 12, "middle",
                 "box " + std::to_string(n.leafIndex));
    }
}

static int max_internal_depth(const std::vector<BVHNode>& nodes, int idx) {
    if (idx < 0 || nodes[idx].leafIndex >= 0) return -1;
    int l = max_internal_depth(nodes, nodes[idx].left);
    int r = max_internal_depth(nodes, nodes[idx].right);
    return 1 + std::max(l, r);
}

// ---- Console print ---------------------------------------------------------

static void print_bvh(const std::vector<BVHNode>& nodes, int root) {
    std::queue<std::pair<int,int>> q;
    q.push({root, 0});
    int cur_depth = -1;
    while (!q.empty()) {
        auto [idx, depth] = q.front(); q.pop();
        const BVHNode& n = nodes[idx];
        if (depth != cur_depth) {
            cur_depth = depth;
            if (n.leafIndex >= 0)
                std::cout << "\n--- Leaves ---\n";
            else
                std::cout << "\n--- Depth " << depth << " ---\n";
        }
        if (n.leafIndex >= 0) {
            std::cout << "  Leaf  node " << idx
                      << "  box " << n.leafIndex
                      << "  bbox [(" << n.bbox.min.x << "," << n.bbox.min.y << ")"
                      << " -> (" << n.bbox.max.x << "," << n.bbox.max.y << ")]\n";
        } else {
            std::cout << "  Inner node " << idx
                      << "  bbox [(" << n.bbox.min.x << "," << n.bbox.min.y << ")"
                      << " -> (" << n.bbox.max.x << "," << n.bbox.max.y << ")]"
                      << "  left=" << n.left << " right=" << n.right << "\n";
            q.push({n.left,  depth + 1});
            q.push({n.right, depth + 1});
        }
    }
    std::cout << '\n';
}

// ---- Main ------------------------------------------------------------------

int main() {
    std::vector<AABB> boxes = {
        AABB(Vec2(0,0), Vec2(1,1)),
        AABB(Vec2(2,1), Vec2(5,2)),
        AABB(Vec2(2,0), Vec2(3,3)),
        AABB(Vec2(4,1), Vec2(5,3)),
    };

    std::vector<BVHNode> nodes;
    int root = build_bvh(boxes, nodes);
    int mid  = max_internal_depth(nodes, root);

    print_bvh(nodes, root);

    const double LEGEND_W = 220;
    SCALE  = 100.0;
    PAD    = 50.0;
    WMIN_X = nodes[root].bbox.min.x;
    WMIN_Y = nodes[root].bbox.min.y;
    WMAX_Y = nodes[root].bbox.max.y;
    double wmax_x = nodes[root].bbox.max.x;

    double svg_w = (wmax_x  - WMIN_X) * SCALE + 2*PAD + LEGEND_W;
    double svg_h = (WMAX_Y - WMIN_Y) * SCALE + 2*PAD + 20; // +20 for x-axis labels

    SVG.open("bvh_out.svg");
    SVG << "<svg xmlns='http://www.w3.org/2000/svg'"
        << " width='" << svg_w << "' height='" << svg_h << "'>\n";
    SVG << "  <rect width='100%' height='100%' fill='#12121f'/>\n";

    draw_grid(WMIN_X, wmax_x, WMIN_Y, WMAX_Y);
    draw_bvh(nodes, root);   // internal nodes, outermost-first (BFS)
    draw_leaves(nodes);      // leaves always on top
    draw_legend(mid, (wmax_x - WMIN_X) * SCALE + 2*PAD, svg_h);

    SVG << "</svg>\n";
    SVG.close();
    std::cout << "Wrote bvh_out.svg — open it in a browser.\n";
    return 0;
}
