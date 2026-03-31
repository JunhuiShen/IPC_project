#pragma once

#include "IPC_math.h"
#include <array>
#include <string>

enum class NodeTriangleRegion{
    FaceInterior, Edge12, Edge23, Edge31, Vertex1, Vertex2, Vertex3, DegenerateTriangle
};

std::string to_string(NodeTriangleRegion region);

struct NodeTriangleDistanceResult{
    Vec3 closest_point;
    Vec3 tilde_x; // projection of x
    Vec3 normal;
    std::array<double, 3> barycentric_tilde_x;
    double phi;
    double distance;
    NodeTriangleRegion region;
};

NodeTriangleDistanceResult node_triangle_distance(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double eps = 1.0e-12);
