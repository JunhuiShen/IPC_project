#pragma once

#include "node_triangle_distance.h"
#include "broad_phase.h"

#include <array>
#include <limits>
#include <vector>

namespace ogc_detail {

// Pair packing and frozen geometry belong to the dedicated OGC path.
inline std::array<Vec3, 4> pair_positions(const NodeTrianglePair& pair, const std::vector<Vec3>& positions) {
    return {positions[pair.node], positions[pair.tri_v[0]], positions[pair.tri_v[1]], positions[pair.tri_v[2]]};
}

inline std::array<Vec3, 4> pair_positions(const SegmentSegmentPair& pair, const std::vector<Vec3>& positions) {
    return {positions[pair.v[0]], positions[pair.v[1]], positions[pair.v[2]], positions[pair.v[3]]};
}

struct ContactCache {
    std::vector<NodeTriangleDistanceResult> nt_distances;
    std::vector<SegmentSegmentDistanceResult> ss_distances;

    void evaluate(const BroadPhase& broad_phase, const std::vector<Vec3>& positions, bool parallel, bool exact_computation_fallback);
    double isotropic_motion_bound(const BroadPhase& broad_phase, int node, double fallback) const;
};

// OGC-only local Newton update: live elasticity and frozen mesh contact.
// Friction parameters/previous positions must already be validated by the solver.
Vec3 frozen_contact_vertex_delta(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& xhat, const std::vector<Vec3>& x_elastic, const std::vector<Vec3>& x_barrier, const BroadPhase& broad_phase, const PinMap* pin_map, const IncidentTriangles* incident_triangles, const std::vector<ShapeGrads>* rest_shape_grads, const std::vector<Vec3>* previous_positions, const ContactCache& contacts);

// Opt in only from OGC. Preserve the shared GS routine and its existing
// face/degenerate policy, but resolve outside-face features geometrically:
// barycentric signs alone do not identify vertex Voronoi regions of obtuse
// triangles. The nearest boundary point is the minimum over all three edges.
inline NodeTriangleDistanceResult node_triangle_distance(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double eps = 1.0e-12) {
    auto result = ::node_triangle_distance(x, x1, x2, x3, eps);
    if (result.region == NodeTriangleRegion::FaceInterior || result.region == NodeTriangleRegion::DegenerateTriangle) return result;

    const std::array<Vec3, 3> vertices{x1, x2, x3};
    constexpr std::array<NodeTriangleRegion, 3> edges{NodeTriangleRegion::Edge12, NodeTriangleRegion::Edge23, NodeTriangleRegion::Edge31};
    constexpr std::array<NodeTriangleRegion, 3> corners{NodeTriangleRegion::Vertex1, NodeTriangleRegion::Vertex2, NodeTriangleRegion::Vertex3};
    double minimum_squared = std::numeric_limits<double>::infinity();
    for (int a = 0; a < 3; ++a) {
        const int b = (a + 1) % 3;
        double t = 0.0;
        const Vec3 closest = segment_closest_point(x, vertices[a], vertices[b], t);
        const double squared = (x - closest).squaredNorm();
        if (squared < minimum_squared) {
            minimum_squared = squared;
            result.closest_point = closest;
            result.region = t <= 0.0 ? corners[a] : t >= 1.0 ? corners[b] : edges[a];
        }
    }
    result.distance = (x - result.closest_point).norm();
    return result;
}

} // namespace ogc_detail
