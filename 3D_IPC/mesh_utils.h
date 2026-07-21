#pragma once

#include "physics.h"

#include <vector>

TriangleDef make_def_triangle(const std::vector<Vec3>& x, const RefMesh& ref_mesh, int tri_idx);

void clear_model(RefMesh& ref_mesh, DeformedState& state, std::vector<Vec2>& X, std::vector<Pin>& pins);

void append_pin(std::vector<Pin>& pins, int vertex_index, const std::vector<Vec3>& x);

// Maps each vertex to {triangle_index, local_node_index} pairs.
// The local_node_index (0,1,2) is stored to avoid searching at call sites.
VertexTriangleMap build_incident_triangle_map(const std::vector<int>& indices);
