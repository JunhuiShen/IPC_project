#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>

// ============================================================
// Structure
// ============================================================
struct Vec2 {
    double x, y;
};

struct AABB2 {
    Vec2 min;
    Vec2 max;
};

enum ObjectType { NODE, SEGMENT };

struct Object {
    AABB2 box;      // Axis-aligned bounding box
    int id;         // Identifier
    ObjectType type;// Node or Segment
};

// ============================================================
// Box information
// ============================================================
inline AABB2 sweptAABB(const AABB2& b, const Vec2& v) {
    AABB2 out{};
    out.min.x = std::min(b.min.x, b.min.x + v.x);
    out.min.y = std::min(b.min.y, b.min.y + v.y);
    out.max.x = std::max(b.max.x, b.max.x + v.x);
    out.max.y = std::max(b.max.y, b.max.y + v.y);
    return out;
}

// ============================================================
// Overlap in y-direction
// ============================================================
inline bool overlap_y(const AABB2& A, const AABB2& B) {
    return !(A.max.y < B.min.y || A.min.y > B.max.y);
}

// ============================================================
// Broad-phase AABB
// ============================================================
template <typename Callback>
void broad_phase_ccd(std::vector<Object>& objects, Callback report) {
    const int n = static_cast<int>(objects.size());

    // Sort by x_min
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return objects[a].box.min.x < objects[b].box.min.x;
    });

    // Make a list A
    std::vector<int> A;

    // For each Bi in sorted order
    for (int idx_i : order) {
        const AABB2& Bi = objects[idx_i].box;

        // Remove from A any box whose xmax < Bi.xmin
        A.erase(std::remove_if(A.begin(), A.end(),[&](int idx_j) {
            const AABB2& Bj = objects[idx_j].box;
            return Bj.max.x < Bi.min.x;
        }), A.end() );

        // Test against remaining boxes in the list
        for (int idx_j : A) {
            const AABB2& Bj = objects[idx_j].box;

            // Skip same-type pairs
            if (objects[idx_i].type == objects[idx_j].type)
                continue;

            // Check overlaps in y-direction
            if (overlap_y(Bi, Bj)) {
                report(objects[idx_i], objects[idx_j]);
            }
        }

        // Add Bi to the list
        A.push_back(idx_i);
    }
}

// ============================================================
// Numerical experiment
// ============================================================
int main() {
    struct DemoObj { AABB2 box; Vec2 vel; };

    // Define node boxes
    std::vector<DemoObj> nodes = {
            { {{0, 0}, {3, 3}}, {0,0} },
            { {{3.5, 2.5}, {7, 5}}, {0,0} }
    };

    // Define segment boxes
    std::vector<DemoObj> segments = {
            { {{4, 0}, {4, 2}}, {0,0} },
            { {{6, 2}, {8, 4}}, {0,0} }
    };

    // Combine into unified list
    std::vector<Object> objects;
    objects.push_back({
        sweptAABB(nodes[0].box, nodes[0].vel),  1, NODE
    });

    objects.push_back({
        sweptAABB(segments[0].box, segments[0].vel), 2, SEGMENT
    });

    objects.push_back({
        sweptAABB(nodes[1].box, nodes[1].vel), 3, NODE
    });

    objects.push_back({
        sweptAABB(segments[1].box, segments[1].vel), 4, SEGMENT
    });

    // Run broad-phase AABB
    broad_phase_ccd(objects, [](const Object& A, const Object& B) {
        std::string tA = (A.type == NODE) ? "Node" : "Segment";
        std::string tB = (B.type == NODE) ? "Node" : "Segment";
        std::cout << "Overlap: " << tA << " " << A.id << " and " << tB << " " << B.id << "\n";
    });

    return 0;
}