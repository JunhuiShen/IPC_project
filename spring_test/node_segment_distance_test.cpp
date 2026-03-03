#include <iostream>
#include <cmath>
#include <algorithm>

struct Vec2 {
    double x, y;
};

// =====================================================
// Function under test
// =====================================================
// Compute point–segment distance
double nodeSegmentDistance(const Vec2 &xi,
                           const Vec2 &xj,
                           const Vec2 &xjp1,
                           double &t, Vec2 &p, Vec2 &r)
{
    // Segment direction
    Vec2 seg = { xjp1.x - xj.x, xjp1.y - xj.y };
    double seg_len2 = seg.x * seg.x + seg.y * seg.y;

    // Handle degenerate segment
    if (seg_len2 < 1e-14) {
        t = 0.0;
        p = xj;
        r = { xi.x - p.x, xi.y - p.y };
        return std::sqrt(r.x * r.x + r.y * r.y);
    }

    // Project point onto segment
    Vec2 q = { xi.x - xj.x, xi.y - xj.y };
    double dot = q.x * seg.x + q.y * seg.y;
    t = dot / seg_len2;

    // Clamp to segment
    t = (t < 0.0) ? 0.0 : (t > 1.0 ? 1.0 : t);

    // Closest point
    p = { xj.x + t * seg.x, xj.y + t * seg.y };

    // Vector and distance
    r = { xi.x - p.x, xi.y - p.y };
    return std::sqrt(r.x * r.x + r.y * r.y);
}

// =====================================================
// Helper for nice printing
// =====================================================
void printCase(const char *label, const Vec2 &xi, const Vec2 &xj, const Vec2 &xjp1)
{
    double t; Vec2 p, r;
    double d = nodeSegmentDistance(xi, xj, xjp1, t, p, r);
    std::cout << "---------------------------------------------\n";
    std::cout << label << "\n";
    std::cout << "Point xi = (" << xi.x << ", " << xi.y << ")\n";
    std::cout << "Seg    = [(" << xj.x << ", " << xj.y << ") -> ("
              << xjp1.x << ", " << xjp1.y << ")]\n";
    std::cout << "Closest point p = (" << p.x << ", " << p.y << ")\n";
    std::cout << "Parameter t = " << t << "\n";
    std::cout << "Vector r = (" << r.x << ", " << r.y << ")\n";
    std::cout << "Distance = " << d << "\n";
}

// =====================================================
// Main test routine
// =====================================================
int main()
{
    // Case 1: point directly left of vertical segment
    printCase("Case 1: vertical segment, point left",
              {0.0, 0.0}, {1.0, -1.0}, {1.0, 1.0});

    // Case 2: point above a horizontal segment
    printCase("Case 2: horizontal segment, point above",
              {1.0, 2.0}, {0.0, 0.0}, {3.0, 0.0});

    // Case 3: point beyond segment endpoint (tests clamping)
    printCase("Case 3: point beyond segment",
              {4.0, 1.0}, {0.0, 0.0}, {3.0, 0.0});

    // Case 4: diagonal segment, off-axis point
    printCase("Case 4: diagonal segment",
              {1.0, 2.0}, {0.0, 0.0}, {2.0, 2.0});

    // Case 5: degenerate (zero-length) segment
    printCase("Case 5: zero-length segment",
              {2.0, 2.0}, {1.0, 1.0}, {1.0, 1.0});

    return 0;
}
