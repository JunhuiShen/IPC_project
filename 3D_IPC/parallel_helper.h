#pragma once
#include "physics.h"
#include "broad_phase.h"
#include <vector>

struct JacobiPrediction {
    bool active = true;
    Vec3 g = Vec3::Zero();
    Vec3 delta = Vec3::Zero();
    AABB certified_region;
};

struct ParallelCommit {
    int vi = -1;
    Vec3 delta = Vec3::Zero();
    double alpha_clip = 1.0;
    double ccd_step = 1.0;
    Vec3 x_after = Vec3::Zero();
    bool valid = false;
};

void build_jacobi_predictions(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                              const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                              const BroadPhase::Cache& bp_cache, std::vector<JacobiPrediction>& predictions,
                              const PinMap* pin_map = nullptr);

// Persistent swept-region BVH; reused across build_conflict_graph calls to
// refit in place of a full rebuild when the active set is stable.
struct SweptBvhCache {
    std::vector<BVHNode> nodes;
    std::vector<int>     active_ids;
    std::vector<AABB>    active_boxes;
    int                  root  = -1;
    bool                 valid = false;
};

// Mesh-adjacency edges of the conflict graph. Invariant for a fixed mesh.
std::vector<std::vector<int>> build_elastic_adj(const RefMesh& ref_mesh, const VertexTriangleMap& adj, int nv);

// Contact edges of the conflict graph from broad-phase NT/SS pair lists.
// Stable between broad-phase refreshes (key on BroadPhase::version()).
std::vector<std::vector<int>> build_contact_adj(const BroadPhase::Cache& bp_cache, int nv);

// Sorted per-vertex union of two sorted neighbor lists.
std::vector<std::vector<int>> union_adjacency(const std::vector<std::vector<int>>& a,
                                              const std::vector<std::vector<int>>& b);

// When `base_adj` is non-null it is used verbatim for the elastic + barrier
// edges and only swept-region edges are added; `sw_cache` enables BVH refit.
std::vector<std::vector<int>> build_conflict_graph(const RefMesh& ref_mesh, const std::vector<Pin>& pins,
                                                   const BroadPhase::Cache& bp_cache, const std::vector<JacobiPrediction>& predictions,
                                                   const VertexTriangleMap* adj = nullptr,
                                                   const std::vector<std::vector<int>>* base_adj = nullptr,
                                                   SweptBvhCache* sw_cache = nullptr);

std::vector<std::vector<int>> greedy_color_conflict_graph(const std::vector<std::vector<int>>& graph, const std::vector<JacobiPrediction>& predictions);

ParallelCommit compute_parallel_commit_for_vertex(int vi, bool use_cached_prediction, const JacobiPrediction& prediction,
                                                  const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                                                  const SimParams& params, const std::vector<Vec3>& x_current, const std::vector<Vec3>& xhat,
                                                  const BroadPhase& broad_phase, const PinMap* pin_map = nullptr);

double compute_safe_step_for_vertex(int vi, const RefMesh& ref_mesh, const SimParams& params,
                                    const std::vector<Vec3>& x, const Vec3& delta,
                                    const BroadPhase& broad_phase);

void apply_parallel_commits(const std::vector<ParallelCommit>& commits, std::vector<Vec3>& xnew);
