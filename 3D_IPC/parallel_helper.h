#pragma once
#include "physics.h"
#include "broad_phase.h"
#include <vector>

// Mesh-adjacency edges of the conflict graph. Invariant for a fixed mesh.
std::vector<std::vector<int>> build_elastic_adj(const RefMesh& ref_mesh, const VertexTriangleMap& adj, int num_vertices);

// Contact edges of the conflict graph from a BroadPhase-generated cache.
void build_contact_adj(const BroadPhase::Cache& bp_cache, int num_vertices, std::vector<std::vector<int>>& out);

// Sorted per-vertex union of two sorted neighbor lists.
void union_adjacency(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b,  std::vector<std::vector<int>>& out);

void greedy_color_conflict_graph(const std::vector<std::vector<int>>& graph, std::vector<std::vector<int>>& groups);
