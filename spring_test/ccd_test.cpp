#include <iostream>
#include <cmath>
#include <iomanip>

struct Vec2 {
    double x, y;
};

// ============================================================
// Continuous Collision Detection (CCD) for a node vs. fixed segment
// ============================================================

double ccd_node_segment_update(const Vec2 &x, const Vec2& dx, const Vec2& x1, const Vec2& x2, double eta = 0.9){
    // Compute segment vector and the squared length
    Vec2 seg = { x2.x - x1.x, x2.y - x1.y };
    double seg_len2 = seg.x * seg.x + seg.y * seg.y;

    // Projection onto infinite line
    double s_proj = ((x.x - x1.x) * seg.x + (x.y - x1.y) * seg.y) / seg_len2;
    Vec2 x_tilde = { x1.x + s_proj * seg.x, x1.y + s_proj * seg.y };

    // Direction from x to the closest point on the segment
    double step_len = std::sqrt(dx.x * dx.x + dx.y * dx.y);
    Vec2 w_tilde = {(x_tilde.x - x.x) / step_len, (x_tilde.y - x.y) / step_len};
    double w_tilde_len = std::sqrt(w_tilde.x * w_tilde.x + w_tilde.y * w_tilde.y);

    // Step-size rule
    double omega_hat;
    if (w_tilde_len > 1.0) {
        omega_hat = 1.0; // line farther than one step
    } else {
        double s = s_proj; // coordinate along segment
        if (s >= 0.0 && s <= 1.0)
            omega_hat = eta * w_tilde_len; // projection within segment
        else
            omega_hat = 1.0; // outside segment → full step
    }

    // Clamp to valid range [0,1]
    if (omega_hat < 0.0) omega_hat = 0.0;
    if (omega_hat > 1.0) omega_hat = 1.0;

    return omega_hat;
}


// ============================================================
// Utility
// ============================================================
double nodeSegmentDistance(const Vec2 &xi, const Vec2 &xj, const Vec2 &xjp1, double &t, Vec2 &p, Vec2 &r){
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

double nodeSegmentSignedDistance(const Vec2 &xi, const Vec2 &xj, const Vec2 &xjp1, double &t, Vec2 &p, Vec2 &r){

    // Segment direction
    Vec2 seg = { xjp1.x - xj.x, xjp1.y - xj.y };
    double seg_len = std::sqrt(seg.x*seg.x + seg.y*seg.y);
    Vec2 n = { -seg.y / seg_len, seg.x / seg_len };
    return n.x * (xi.x - xj.x) + n.y * (xi.y - xj.y);
}

void printVec(const std::string &name, const Vec2 &v) {
    std::cout << name << " = (" << std::setw(12) << v.x << ", " << std::setw(12) << v.y << ")\n";
}

// ============================================================
// Test 1: Generic node-segment crossing
// ============================================================
void test_generic_case() {
    std::cout << "\n=============================\n";
    std::cout << " Test 1: Generic CCD example\n";
    std::cout << "=============================\n";

    Vec2 x  = { 0.5,  0.0 };
    Vec2 dx = {-0.5, -1.0 };
    Vec2 x1 = {-1.0,  0.0 };
    Vec2 x2 = { 0.0, -1.0 };
    double eta = 0.9;

    printVec("x", x);
    printVec("dx", dx);
    printVec("x1", x1);
    printVec("x2", x2);

    double omega = ccd_node_segment_update(x, dx, x1, x2, eta);
    std::cout << "omega = " << omega << "\n";

    Vec2 x_next = { x.x + omega * dx.x, x.y + omega * dx.y };
    printVec("x_next (CCD)", x_next);
    printVec("x_full (no CCD)", { x.x + dx.x, x.y + dx.y });
}


// ============================================================
// Main Entry Point
// ============================================================
int main() {
    std::cout << std::fixed << std::setprecision(9);
    test_generic_case();
    return 0;
}