#include "node_triangle_distance.h"

#include <cmath>
#include <limits>

std::string to_string(NodeTriangleRegion region){
    switch (region) {
        case NodeTriangleRegion::FaceInterior: return "FaceInterior";
        case NodeTriangleRegion::Edge12: return "Edge12";
        case NodeTriangleRegion::Edge23: return "Edge23";
        case NodeTriangleRegion::Edge31: return "Edge31";
        case NodeTriangleRegion::Vertex1: return "Vertex1";
        case NodeTriangleRegion::Vertex2: return "Vertex2";
        case NodeTriangleRegion::Vertex3: return "Vertex3";
        case NodeTriangleRegion::DegenerateTriangle: return "DegenerateTriangle";
        default: return "Unknown";
    }
}

NodeTriangleDistanceResult node_triangle_distance(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double eps){
    NodeTriangleDistanceResult out;
    out.closest_point = Vec3::Zero();
    out.tilde_x = Vec3::Zero();
    out.normal = Vec3::Zero();
    out.barycentric_tilde_x = {{0.0, 0.0, 0.0}};
    out.phi = 0.0;
    out.distance = 0.0;
    out.region = NodeTriangleRegion::DegenerateTriangle;

    const Vec3 e12 = x2 - x1;
    const Vec3 e13 = x3 - x1;
    const Vec3 n_raw = e12.cross(e13);
    const double n_norm = n_raw.norm();

    if (n_norm <= eps) {
        double t12 = 0.0, t23 = 0.0, t31 = 0.0;
        const Vec3 q12 = segment_closest_point(x, x1, x2, t12);
        const Vec3 q23 = segment_closest_point(x, x2, x3, t23);
        const Vec3 q31 = segment_closest_point(x, x3, x1, t31);

        const double d12 = (x - q12).norm();
        const double d23 = (x - q23).norm();
        const double d31 = (x - q31).norm();

        if (d12 <= d23 && d12 <= d31) {
            out.closest_point = q12;
            out.distance = d12;
        } else if (d23 <= d12 && d23 <= d31) {
            out.closest_point = q23;
            out.distance = d23;
        } else {
            out.closest_point = q31;
            out.distance = d31;
        }

        out.tilde_x = out.closest_point;
        return out;
    }

    const Vec3 n = n_raw / n_norm;
    const double phi = n.dot(x - x1);
    const Vec3 tilde_x = x - phi * n;

    out.normal = n;
    out.phi = phi;
    out.tilde_x = tilde_x;

    const auto lambda = triangle_plane_barycentric_coordinates(tilde_x, x1, x2, x3, eps);
    out.barycentric_tilde_x = lambda;

    const double l1 = lambda[0];
    const double l2 = lambda[1];
    const double l3 = lambda[2];

    if (l1 >= 0.0 && l2 >= 0.0 && l3 >= 0.0) {
        out.closest_point = tilde_x;
        out.distance = std::abs(phi);
        out.region = NodeTriangleRegion::FaceInterior;
        return out;
    }

    // Vertex regions: two barycentric coordinates <= 0
    // Vertex 1: l2 <= 0 and l3 <= 0
    if (l2 <= 0.0 && l3 <= 0.0) {
        out.closest_point = x1;
        out.distance = (x - x1).norm();
        out.region = NodeTriangleRegion::Vertex1;
        return out;
    }

    // Vertex 2: l3 <= 0 and l1 <= 0
    if (l3 <= 0.0 && l1 <= 0.0) {
        out.closest_point = x2;
        out.distance = (x - x2).norm();
        out.region = NodeTriangleRegion::Vertex2;
        return out;
    }

    // Vertex 3: l1 <= 0 and l2 <= 0
    if (l1 <= 0.0 && l2 <= 0.0) {
        out.closest_point = x3;
        out.distance = (x - x3).norm();
        out.region = NodeTriangleRegion::Vertex3;
        return out;
    }

    // Edge regions: exactly one barycentric coordinate < 0
    if (l3 < 0.0) {
        double t = 0.0;
        out.closest_point = segment_closest_point(tilde_x, x1, x2, t);
        out.distance = (x - out.closest_point).norm();
        out.region = NodeTriangleRegion::Edge12;
        return out;
    }

    if (l1 < 0.0) {
        double t = 0.0;
        out.closest_point = segment_closest_point(tilde_x, x2, x3, t);
        out.distance = (x - out.closest_point).norm();
        out.region = NodeTriangleRegion::Edge23;
        return out;
    }

    if (l2 < 0.0) {
        double t = 0.0;
        out.closest_point = segment_closest_point(tilde_x, x3, x1, t);
        out.distance = (x - out.closest_point).norm();
        out.region = NodeTriangleRegion::Edge31;
        return out;
    }

    // numerical fallback
    {
        double t12 = 0.0, t23 = 0.0, t31 = 0.0;
        const Vec3 q12 = segment_closest_point(tilde_x, x1, x2, t12);
        const Vec3 q23 = segment_closest_point(tilde_x, x2, x3, t23);
        const Vec3 q31 = segment_closest_point(tilde_x, x3, x1, t31);

        const double d12 = (x - q12).norm();
        const double d23 = (x - q23).norm();
        const double d31 = (x - q31).norm();

        if (d12 <= d23 && d12 <= d31) {
            out.closest_point = q12;
            out.distance = d12;
            out.region = NodeTriangleRegion::Edge12;
        } else if (d23 <= d12 && d23 <= d31) {
            out.closest_point = q23;
            out.distance = d23;
            out.region = NodeTriangleRegion::Edge23;
        } else {
            out.closest_point = q31;
            out.distance = d31;
            out.region = NodeTriangleRegion::Edge31;
        }
    }

    return out;
}
