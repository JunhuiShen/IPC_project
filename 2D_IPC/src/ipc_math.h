#pragma once

#include <cmath>
#include <vector>

// ======================================================
// Core types
// ======================================================

struct Vec2 {
    double x = 0, y = 0;
    Vec2() = default;
    Vec2(double x_, double y_) : x(x_), y(y_) {}
    Vec2 operator+(const Vec2 &o) const { return Vec2(x + o.x, y + o.y); }
    Vec2 operator-(const Vec2 &o) const { return Vec2(x - o.x, y - o.y); }
    Vec2 operator*(double s)      const { return Vec2(x * s,   y * s);   }
};

using Vec = std::vector<double>;

struct Mat2 {
    double a11, a12, a21, a22;
};

// Rank-3 tensor, stored as two 2x2 matrices: [m0, m1]
struct Mat2x2x2 {
    Mat2 m0, m1;
};

// Get position of node i from flat position vector
inline Vec2 get_xi(const Vec &x, int i) {
    return {x[2 * i], x[2 * i + 1]};
}

// Set position of node i in flat position vector
inline void set_xi(Vec &x, int i, const Vec2 &v) {
    x[2 * i]     = v.x;
    x[2 * i + 1] = v.y;
}

// ======================================================
// Math operations
// ======================================================

namespace math {

    inline double node_distance(const Vec &a, int i, int j) {
        double dx = a[2 * i]     - a[2 * j];
        double dy = a[2 * i + 1] - a[2 * j + 1];
        return std::sqrt(dx * dx + dy * dy);
    }

    // --- Vec2 ---

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

    static inline double cross(const Vec2 &a, const Vec2 &b) {
        return a.x * b.y - a.y * b.x;
    }

    static inline double norm2(const Vec2 &a) {
        return dot(a, a);
    }

    static inline double norm(const Vec2 &a) {
        return std::sqrt(norm2(a));
    }

    // --- Mat2 ---

    static inline Mat2 outer(const Vec2 &a, const Vec2 &b) {
        return {a.x * b.x, a.x * b.y, a.y * b.x, a.y * b.y};
    }

    static inline Mat2 matmul(const Mat2 &A, const Mat2 &B) {
        return {
            A.a11 * B.a11 + A.a12 * B.a21, A.a11 * B.a12 + A.a12 * B.a22,
            A.a21 * B.a11 + A.a22 * B.a21, A.a21 * B.a12 + A.a22 * B.a22
        };
    }

    static inline Mat2 add(const Mat2 &A, const Mat2 &B) {
        return {A.a11 + B.a11, A.a12 + B.a12, A.a21 + B.a21, A.a22 + B.a22};
    }

    static inline Mat2 scale(const Mat2 &A, double s) {
        return {s * A.a11, s * A.a12, s * A.a21, s * A.a22};
    }

    static inline Mat2 transpose(const Mat2 &M) {
        return {M.a11, M.a21, M.a12, M.a22};
    }

    static inline Vec2 matvec(const Mat2 &A, const Vec2 &v) {
        return {A.a11 * v.x + A.a12 * v.y, A.a21 * v.x + A.a22 * v.y};
    }

} // namespace math
