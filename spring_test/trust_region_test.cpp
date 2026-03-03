#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
using std::isfinite;

// --- Core data structure ---
struct Vec2 {
    double x, y;
};

// Core math and linear algebra
namespace math {
    static inline Vec2 add(const Vec2 &a, const Vec2 &b) {
        return {a.x + b.x, a.y + b.y};
    }

    static inline Vec2 sub(const Vec2 &a, const Vec2 &b) {
        return {a.x - b.x, a.y - b.y};
    }

    static inline Vec2 scale(const Vec2 &a, double s) {
        return {s * a.x, s * a.y};
    }

    static inline double dot(const Vec2 &a, const Vec2 &b) {
        return a.x * b.x + a.y * b.y;
    }

    static inline double norm2(const Vec2 &a) {
        return a.x * a.x + a.y * a.y;
    }

    static inline double norm(const Vec2 &a)  {
        return std::sqrt(norm2(a));
    }
}

// -----------------------------------------------------------------------------
// Compute node–segment distance (UNCHANGED)
// -----------------------------------------------------------------------------
double node_segment_distance(const Vec2 &xi, const Vec2 &xj, const Vec2 &xjp1,
                             double &t, Vec2 &p, Vec2 &r)
{
    Vec2 seg = {xjp1.x - xj.x, xjp1.y - xj.y};
    double seg_len2 = seg.x * seg.x + seg.y * seg.y;

    if (seg_len2 < 1e-14) {
        t = 0.0;
        p = xj;
        r = math::sub(xi, p);
        return math::norm(r);
    }

    Vec2 q = math::sub(xi, xj);
    double proj = math::dot(q, seg);
    t = proj / seg_len2;

    t = (t < 0.0) ? 0.0 : (t > 1.0 ? 1.0 : t);

    p = math::add(xj, math::scale(seg, t));
    r = math::sub(xi, p);
    return math::norm(r);
}

// -----------------------------------------------------------------------------
// Standalone Trust-Region Weight (fully dynamic case)
// -----------------------------------------------------------------------------
static double trust_region_weight(
        const Vec2& x,  const Vec2& dx,
        const Vec2& x2, const Vec2& dx2,
        const Vec2& x3, const Vec2& dx3,
        double eta)
{
    double s_closest;
    Vec2 p{}, r{};
    double d0 = node_segment_distance(x, x2, x3, s_closest, p, r);

    double M = math::norm(dx) + math::norm(dx2) + math::norm(dx3);

    double w = (eta * d0) / (M);
    return std::max(0.0, std::min(1.0, w));
}

// -----------------------------------------------------------------------------
// Test reporter
// -----------------------------------------------------------------------------
static void report(const char* name, const Vec2& x,  const Vec2& dx, const Vec2& x2, const Vec2& dx2,
                   const Vec2& x3, const Vec2& dx3, double eta){
    double s0; Vec2 p0{}, r0{};
    double d0 = node_segment_distance(x, x2, x3, s0, p0, r0);
    double M  = math::norm(dx) + math::norm(dx2) + math::norm(dx3);
    double w  = trust_region_weight(x, dx, x2, dx2, x3, dx3, eta);

    std::cout << "== " << name << " ==\n";
    std::cout << "d0 = " << d0
              << "   M = " << M
              << "   w_safe = " << w << "\n\n";
}

int main() {
    const double eta = 0.9;

    report("Test 1: fixed segment",
           {0,1},{0,-1}, {-1,0},{0,0}, {1,0},{0,0}, eta);

    report("Test 2: parallel motion",
           {0,1},{0,-1}, {-1,0},{0,-1}, {1,0},{0,-1}, eta);

    report("Test 3: one endpoint moves",
           {0,1},{0,-1}, {-1,0},{0,0}, {1,0},{0.5,0.0}, eta);

    report("Test 4: fully dynamic",
           {0,1},{0.1,-1.0}, {-1,0},{0.3,0.0}, {1,0},{0.2,0.5}, eta);

    // Test 5: Large displacement toward vertical segment
    report("Test 5: Large displacement toward vertical segment",
           {-1, 0}, {5.0, 0.0},   // x1 moves strongly to the right
           {0,  1}, {0.0, 0.0},   // fixed segment
           {0, -1}, {0.0, 0.0},   // fixed segment
           eta);

    // -------------------------------------------------------------------------
    // Test 6: Two chains, each 11 nodes (10 segments).
    // Right chain starts VERY close to left chain, and moves left.
    // Compute weight for every right node:
    //   weight(node r) = min over all left segments s of trust_region_weight(node r, seg s)
    // -------------------------------------------------------------------------
    {
        std::cout << "== Test 6: two 11-node chains, right VERY close -> left, weights for right nodes ==\n";

        constexpr int N = 11;
        std::vector<Vec2> left_x(N), left_dx(N);
        std::vector<Vec2> right_x(N), right_dx(N);

        const double gap = 1e-3;   // <-- very close (try 1e-6, 1e-2, etc.)
        const double v   = -6.0;   // moving left

        for (int i = 0; i < N; ++i) {
            double y = -5.0 + double(i);

            left_x[i]  = {0.0, y};
            left_dx[i] = {0.0, 0.0};

            right_x[i]  = {gap, y};      // <-- close to x=0
            right_dx[i] = {v, 0.0};      // <-- moving left (toward/through left chain)
        }

        for (int r = 0; r < N; ++r) {
            double w_min = 1.0;

            for (int s = 0; s < N - 1; ++s) {
                double w = trust_region_weight(
                        right_x[r],  right_dx[r],
                        left_x[s],   left_dx[s],
                        left_x[s+1], left_dx[s+1],
                        eta
                );
                w_min = std::min(w_min, w);
            }

            std::cout << "right node " << r
                      << " pos=(" << right_x[r].x << "," << right_x[r].y << ")"
                      << "  w_min=" << w_min << "\n";
        }

        std::cout << "\n";
    }

    return 0;
}