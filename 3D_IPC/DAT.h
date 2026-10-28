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
