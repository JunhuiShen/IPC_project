#pragma once
#include "physics.h"
#include "broad_phase.h"
#include <vector>

// Exact AABB of a rigid node's spherical-cap rotation envelope.
AABB spherical_cap_node_aabb(const Vec3& x_com, const Vec4& q, const Vec3& X, double theta_bound);

// Rigid-node blue boxes: spherical-cap rotation plus padded COM translation.
void build_blue_boxes_rb(const std::vector<Vec3>& com_box_anchors, const std::vector<Vec4>& orientation_box_anchors, const std::vector<double>& theta_box_radii, const std::vector<double>& com_box_radii, const RefMesh& ref_mesh, std::vector<AABB>& blue_boxes);

int owning_rb_for_node(const std::vector<int>& node_to_rb, int node);

std::vector<int> build_node_to_block(const std::vector<int>& node_to_rb, const std::vector<int>& deformable_nodes, int num_rbs);

// Per-body broad-phase contacts and their rigid-body conflict graph.
void build_rb_contact_adj(const BroadPhase::Cache& bp_cache, const std::vector<int>& node_to_rb, int num_rbs, std::vector<std::vector<int>>& body_nt_pair_indices, std::vector<std::vector<int>>& body_ss_pair_indices, std::vector<std::vector<int>>& out);

// Mixed graphs use cached blocks [0, num_deformable) for cloth nodes, followed
// by one block per rigid body. node_to_block is built once per topology.
void build_block_elastic_adj(const std::vector<std::vector<int>>& nodal_elastic_adj, const std::vector<int>& node_to_block, const std::vector<std::vector<int>>& block_nodes, std::vector<std::vector<int>>& out);

// Contact conflicts in mixed block indexing. Every node-triangle and
// segment-segment candidate contributes a clique among its distinct cloth-node
// and rigid-body blocks, covering cloth-cloth, cloth-rigid, and rigid-rigid
// contacts in one graph.
// Parallel and fused: each block owns one graph row, and rigid blocks also
// build their NT/SS pair lists during the same incidence scan.
void build_block_contact_adj(const BroadPhase::Cache& bp_cache, const std::vector<int>& node_to_block, const std::vector<std::vector<int>>& block_nodes, int num_deformable_blocks, std::vector<std::vector<int>>& body_nt_pair_indices, std::vector<std::vector<int>>& body_ss_pair_indices, std::vector<std::vector<int>>& out);

// Mesh-adjacency edges of the conflict graph. Invariant for a fixed mesh.
std::vector<std::vector<int>> build_elastic_adj(const RefMesh& ref_mesh, const VertexTriangleMap& adj, int num_vertices);

// Contact edges of the conflict graph from a BroadPhase-generated cache.
void build_contact_adj(const BroadPhase::Cache& bp_cache, int num_vertices, std::vector<std::vector<int>>& out);

// Sorted per-vertex union of two sorted neighbor lists.
void union_adjacency(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b,  std::vector<std::vector<int>>& out);

void greedy_color_conflict_graph(const std::vector<std::vector<int>>& graph, std::vector<std::vector<int>>& groups);
