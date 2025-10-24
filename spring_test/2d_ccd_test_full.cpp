#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

struct Vec2 {
    double x, y;
};

// -----------------------------------------------------------------------------
// 2D Point–Segment CCD
// -----------------------------------------------------------------------------
bool ccd_point_segment_2d(const Vec2& x1, const Vec2& dx1, const Vec2& x2, const Vec2& dx2,
                          const Vec2& x3, const Vec2& dx3, double& t_out, double eps = 1e-12){

    // Vector computation
    auto sub = [](const Vec2& a, const Vec2& b) -> Vec2 {
        return {a.x - b.x, a.y - b.y};
    };
    auto add = [](const Vec2& a, const Vec2& b) -> Vec2 {
        return {a.x + b.x, a.y + b.y};
    };
    auto mul = [](const Vec2& a, double s) -> Vec2 {
        return {a.x * s, a.y * s};
    };
    auto dot = [](const Vec2& a, const Vec2& b) -> double {
        return a.x * b.x + a.y * b.y;
    };
    auto cross = [](const Vec2& a, const Vec2& b) -> double {
        return a.x * b.y - a.y * b.x;
    };
    auto norm2 = [&](const Vec2& a) -> double {
        return dot(a, a);
    };

    // Compute coefficients of f(t) = a t^2 + b t + c
    Vec2 x21  = sub(x1, x2);
    Vec2 x32  = sub(x3, x2);
    Vec2 dx21 = sub(dx1, dx2);
    Vec2 dx32 = sub(dx3, dx2);

    double a = cross(dx32, dx21);
    double b = cross(dx32, x21) + cross(x32, dx21);
    double c = cross(x32, x21);

    double t_candidates[2];
    int num_roots = 0;

    // Degenerate case if a = 0
    if (std::fabs(a) < eps) {
        if (std::fabs(b) < eps) return false;
        double t = -c / b;
        if (t >= 0.0 && t <= 1.0)
            t_candidates[num_roots++] = t;
    }
    else {
        double D = b * b - 4.0 * a * c;
        if (D < 0.0) return false; // No real roots

        double sqrtD = std::sqrt(std::max(D, 0.0));
        double s = (b >= 0.0) ? 1.0 : -1.0;
        double q = -0.5 * (b + s * sqrtD);

        double t1 = q / a;
        double t2 = c / q;

        if (t1 >= 0.0 && t1 <= 1.0)
            t_candidates[num_roots++] = t1;
        if (t2 >= 0.0 && t2 <= 1.0)
            t_candidates[num_roots++] = t2;
    }

    if (num_roots == 0) return false;

    // Choose earliest valid collision time
    double t_star = t_candidates[0];
    if (num_roots == 2 && t_candidates[1] < t_star)
        t_star = t_candidates[1];

    // Inside-segment test
    Vec2 x1t = add(x1, mul(dx1, t_star));
    Vec2 x2t = add(x2, mul(dx2, t_star));
    Vec2 x3t = add(x3, mul(dx3, t_star));

    Vec2 seg = sub(x3t, x2t);
    Vec2 rel = sub(x1t, x2t);

    double seg_len2 = norm2(seg);
    if (seg_len2 < eps) return false; // Degenerate segment

    double s = dot(rel, seg) / seg_len2;
    if (s < 0.0 || s > 1.0) return false;

    // Valid collision
    t_out = t_star;
    return true;
}

double ccd_get_safe_step(const Vec2& x1, const Vec2& dx1, const Vec2& x2, const Vec2& dx2,
                         const Vec2& x3, const Vec2& dx3, double eta = 0.9) {

    double t_hit; // Variable to store the time of impact

    // Run the 2D CCD to find the exact time of impact.
    bool collision_found = ccd_point_segment_2d(x1, dx1, x2, dx2, x3, dx3, t_hit);

    if (collision_found) {
        if (t_hit <= 1e-12) {
            // Already in collision, don't move at all
            return 0.0;
        }
        return eta * t_hit;
    } else {
        // No collision found in the [0, 1] interval
        return 1.0;
    }
}

// Test cases
int main() {
    double t_hit;

    // Dynamic node vs fixed segment
    {
        Vec2 x1{0, 1}, dx1{0, -1};
        Vec2 x2{-1, 0}, dx2{0, 0};
        Vec2 x3{ 1, 0}, dx3{0, 0};

        std::cout << "Test 1: Dynamic node vs fixed segment" << std::endl;
        if (ccd_point_segment_2d(x1, dx1, x2, dx2, x3, dx3, t_hit))
            std::cout << "Collision at t = " << t_hit << "\n\n";
        else
            std::cout << "No collision\n\n";
    }

    // Dynamic node vs segment moving in parallel
    {
        Vec2 x1{0, 1}, dx1{0, -1};
        Vec2 x2{-1, 0}, dx2{0, -1};
        Vec2 x3{ 1, 0}, dx3{0, -1};

        std::cout << "Dynamic node and segment moving in parallel" << std::endl;
        if (ccd_point_segment_2d(x1, dx1, x2, dx2, x3, dx3, t_hit))
            std::cout << "Collision at t = " << t_hit << "\n\n";
        else
            std::cout << "No collision\n\n";
    }

    // Dynamic node vs segment with one fixed endpoint
    {
        Vec2 x1{0, 1}, dx1{0, -1};
        Vec2 x2{-1, 0}, dx2{0, 0};
        Vec2 x3{ 1, 0}, dx3{0.5, 0.0};

        std::cout << "Dynamic node vs segment with one fixed endpoint" << std::endl;
        if (ccd_point_segment_2d(x1, dx1, x2, dx2, x3, dx3, t_hit))
            std::cout << "Collision at t = " << t_hit << "\n\n";
        else
            std::cout << "No collision\n\n";
    }

    // Fully dynamic
    {
        Vec2 x1{0, 1}, dx1{0.1, -1.0};
        Vec2 x2{-1, 0}, dx2{0.3, 0.0};
        Vec2 x3{ 1, 0}, dx3{0.2, 0.5};

        std::cout << "Test 4: Fully dynamic" << std::endl;
        if (ccd_point_segment_2d(x1, dx1, x2, dx2, x3, dx3, t_hit))
            std::cout << "Collision at t = " << t_hit << "\n\n";
        else
            std::cout << "No collision\n\n";
    }

    return 0;
}



