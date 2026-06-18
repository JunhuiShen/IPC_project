#pragma once

#include <cmath>
#include <stdexcept>
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

using Vec = std::vector<Vec2>;

struct Mat2 {
    double a11, a12, a21, a22;
};

// Rank-3 tensor, stored as two 2x2 matrices: [m0, m1]
struct Mat2x2x2 {
    Mat2 m0, m1;
};

// Get position of node i from a per-node position vector.
inline Vec2 get_xi(const Vec &x, int i) {
    return x[i];
}

// Set position of node i in a per-node position vector.
inline void set_xi(Vec &x, int i, const Vec2 &v) {
    x[i] = v;
}

// ======================================================
// Math operations
// ======================================================

inline double node_distance(const Vec &a, int i, int j) {
    double dx = a[i].x - a[j].x;
    double dy = a[i].y - a[j].y;
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

static inline double vec_entry(const Vec2& v, int index) {
    return index == 0 ? v.x : v.y;
}

static inline void set_vec_entry(Vec2& v, int index, double value) {
    if (index == 0) {
        v.x = value;
    } else {
        v.y = value;
    }
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

static inline double mat_entry(const Mat2& A, int row, int col) {
    if (row == 0 && col == 0) return A.a11;
    if (row == 0 && col == 1) return A.a12;
    if (row == 1 && col == 0) return A.a21;
    return A.a22;
}

static inline void set_mat_entry(Mat2& A, int row, int col, double value) {
    if (row == 0 && col == 0) {
        A.a11 = value;
    } else if (row == 0 && col == 1) {
        A.a12 = value;
    } else if (row == 1 && col == 0) {
        A.a21 = value;
    } else {
        A.a22 = value;
    }
}

static inline double kronecker_delta(int i, int j) {
    return i == j ? 1.0 : 0.0;
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

static inline Mat2 mat2_inverse(const Mat2& H) {
    double det = H.a11 * H.a22 - H.a12 * H.a21;
    if (std::abs(det) < 1e-12)
        throw std::runtime_error("Singular matrix in mat2_inverse()");
    double inv = 1.0 / det;
    return {H.a22 * inv, -H.a12 * inv, -H.a21 * inv, H.a11 * inv};
}
