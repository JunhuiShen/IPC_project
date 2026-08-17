#pragma once

#include "IPC_math.h"

#include <array>
#include <cstddef>
#include <vector>

// Local vertices of the face opposite local tet vertex f. This is the
// std::array<int, 3> version of TGSL::MESH::TetFace.
std::array<int, 3> TetFace(std::size_t f);

// Validate a flat tetrahedron index array against a set of rest positions.
//
// Every four consecutive indices define one tetrahedron.  This function
// throws std::invalid_argument for malformed connectivity, non-finite rest
// positions, repeated vertices, duplicate tetrahedra, non-positive rest
// orientation, or (near-)degenerate tetrahedra. It throws std::out_of_range
// for invalid vertex indices. relative_degeneracy_tolerance is dimensionless:
// det(Dm) must exceed tolerance * max_edge_length^3, matching TGSL PBGS's
// positive-measure convention.
void validate_tet_mesh(
    const std::vector<int>& tets,
    const std::vector<Vec3>& rest_positions,
    double relative_degeneracy_tolerance = 1e-12);

// TGSL-style boundary extraction from flat tet connectivity. Local faces use
// TetFace, with the same order as TGSL::MESH::TetFace:
//   {1,2,3}, {0,3,2}, {0,1,3}, {0,2,1}.
// Therefore faces point outward when rest tets have positive orientation.
// Shared faces are removed with an orientation-independent key. A face shared
// by more than two tets is rejected as non-manifold.
std::vector<int> compute_boundary_tri_mesh(
    const std::vector<int>& tets);

// TGSL ComputeBoundaryTriMeshNode equivalent. Return each vertex referenced
// by boundary_tris once, preserving its first-appearance order.
std::vector<int> compute_boundary_tri_mesh_nodes(
    const std::vector<int>& boundary_tris);
