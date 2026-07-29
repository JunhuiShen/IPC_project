#pragma once
#include "physics.h"
#include "broad_phase.h"
#include <vector>

// Bound the actual directed rotation of material point X from orientation to
// q_new. q_rel uses the world-relative convention
// q_rel = q_new * orientation^-1, and its raw sign selects the full arc. The
// swept positions are x_com + R(q_rel(t) * orientation) X for t in [0, 1].
AABB arc_node_aabb(
    const Vec3& x_com, const Vec4& orientation,
    const Vec3& X, const Vec4& q_rel);

// Mesh-adjacency edges of the conflict graph. Invariant for a fixed mesh.
std::vector<std::vector<int>> build_elastic_adj(const RefMesh& ref_mesh, const VertexTriangleMap& adj, int num_vertices);

// Contact edges of the conflict graph from a BroadPhase-generated cache.
void build_contact_adj(const BroadPhase::Cache& bp_cache, int num_vertices, std::vector<std::vector<int>>& out);

// Sorted per-vertex union of two sorted neighbor lists.
void union_adjacency(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b,  std::vector<std::vector<int>>& out);

void greedy_color_conflict_graph(const std::vector<std::vector<int>>& graph, std::vector<std::vector<int>>& groups);
