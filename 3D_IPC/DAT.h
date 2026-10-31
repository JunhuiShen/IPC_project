#pragma once

#include "IPC_math.h"

#include <vector>

struct RefMesh;

// Planar Divide and Truncate for simultaneous, straight-line vertex updates.
// Returns displacements; neither input array nor the mesh is modified.
//
// 1. Cap each update's Euclidean norm at 0.5 * gamma * query_radius.
// 2. Build the existing BVH from blue boxes centered at current_positions,
//    with half-extent query_radius * node_box_padding (default padding: 1.2).
// 3. Keep nonincident vertex/triangle and edge/edge pairs at distance < r_q.
// 4. Allocate their separating planes in proportion to approaching motion,
//    then apply the minimum relaxed plane-intersection factor per vertex.
//
// Requires 0 < gamma < 1, finite positive query_radius, padding >= 1, and
// initially separated nonincident primitives. Invalid inputs throw
// std::invalid_argument; unresolved/zero-gap separation throws std::domain_error.
// For solids, mesh.tet_nodes and mesh.surface_nodes must classify interior and
// boundary nodes consistently, as required by BroadPhase::initialize_surface_nodes.
// This is a surface-contact filter: it does not prevent tetrahedron inversion,
// enforce rigid-body motion, handle curved rotational paths, or query SDFs.
// No existing solver is switched to DAT by this standalone function.
std::vector<Vec3> planar_DAT(
    double gamma,
    double query_radius,
    const RefMesh& mesh,
    const std::vector<Vec3>& incremental_update,
    const std::vector<Vec3>& current_positions,
    double node_box_padding = 1.2);

struct GeneralDATResult {
    std::vector<double> body_factors;
    // The mixed path returns the exact represented endpoints/poses certified
    // by the filter, so callers publish them without recomputing arithmetic.
    std::vector<Vec3> positions, centers, omega;
    std::vector<Vec4> orientations;
};

// Mixed planar DAT. Deformables follow straight lines; rigid vertices follow
// c(t)=c+t*dc, q(t)=E(omega+t*domega,dt)*q_n. One factor scales both
// generalized increments of a body. All inputs are immutable sweep snapshots.
GeneralDATResult general_planar_DAT(double gamma, double query_radius,
    const RefMesh& mesh, const std::vector<Vec3>& particle_updates,
    const std::vector<Vec3>& positions, const std::vector<Vec3>& centers,
    const std::vector<Vec3>& omega, const std::vector<Vec4>& previous_orientations,
    const std::vector<Vec3>& delta_centers, const std::vector<Vec3>& delta_omega,
    double dt, double node_box_padding = 1.2);
