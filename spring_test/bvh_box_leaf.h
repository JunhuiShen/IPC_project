#pragma once
#include <vector>

struct Vec2 {
    double x=0, y=0;
    Vec2() = default;
    Vec2(double x_, double y_) : x(x_), y(y_) {}
    Vec2 operator+(const Vec2& o) const { return Vec2(x+o.x, y+o.y); }
    Vec2 operator-(const Vec2& o) const { return Vec2(x-o.x, y-o.y); }
    Vec2 operator*(double s) const { return Vec2(x*s, y*s); }
};

struct AABB {
    Vec2 min, max;
    AABB();
    AABB(const Vec2& a, const Vec2& b);
    void expand(const AABB& o);
    void expand(const Vec2& p);
    Vec2 centroid() const;
    Vec2 extent() const;
};

struct BVHNode {
    AABB bbox;
    int left = -1, right = -1;
    int leafIndex = -1;
};

int  build_bvh(const std::vector<AABB>& boxes, std::vector<BVHNode>& out);
void refit_bvh(std::vector<BVHNode>& nodes, const std::vector<AABB>& boxes);
bool aabb_intersects(const AABB& a, const AABB& b);
void query_bvh   (const std::vector<BVHNode>& nodes, int nodeIdx,
                  const AABB& query, std::vector<int>& hits);
