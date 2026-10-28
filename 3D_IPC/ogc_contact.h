#pragma once

#include "node_triangle_distance.h"
#include "broad_phase.h"

#include <array>
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

} // namespace ogc_detail
