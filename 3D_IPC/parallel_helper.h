#pragma once
#include "physics.h"
#include "broad_phase.h"
#include <vector>

// Mesh-adjacency edges of the conflict graph. Invariant for a fixed mesh.
std::vector<std::vector<int>> build_elastic_adj(const RefMesh& ref_mesh, const VertexTriangleMap& adj, int num_vertices);

// Contact edges of the conflict graph from broad-phase NT/SS pair lists.
// Stable between broad-phase initialize() calls.
struct ContactAdjacencyScratch {
    std::vector<std::vector<std::vector<int>>> local_nbr;
    int threads = 0;
    int vertices = 0;
};

void build_contact_adj(const BroadPhase::Cache& bp_cache, int num_vertices, std::vector<std::vector<int>>& out, ContactAdjacencyScratch* scratch = nullptr);

// Sorted per-vertex union of two sorted neighbor lists.
void union_adjacency(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b,  std::vector<std::vector<int>>& out);

void greedy_color_conflict_graph(const std::vector<std::vector<int>>& graph,
                                 std::vector<std::vector<int>>& groups);

void compute_local_newton_direction(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                                    const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                                    const BroadPhase::Cache& bp_cache, Vec3& g_out, Mat33& H_out, Vec3& delta_out,
                                    const PinMap* pin_map = nullptr,
                                    const std::vector<TriPrecompute>* tri_cache = nullptr,
                                    const std::vector<HingePrecompute>* hinge_cache = nullptr);

double compute_safe_step_for_vertex(int vi, const RefMesh& ref_mesh, const SimParams& params,
                                    const std::vector<Vec3>& x, const Vec3& delta,
                                    const BroadPhase::Cache& bp_cache);
