#pragma once
#include "physics.h"
#include "broad_phase.h"
#include <vector>

struct JacobiPrediction {
    bool active = true;
    Vec3 g = Vec3::Zero();
    Mat33 H = Mat33::Zero();
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

std::vector<std::vector<int>> build_conflict_graph(const RefMesh& ref_mesh, const std::vector<Pin>& pins,
                                                   const BroadPhase::Cache& bp_cache, const std::vector<JacobiPrediction>& predictions,
                                                   const VertexTriangleMap* adj = nullptr);

std::vector<std::vector<int>> greedy_color_conflict_graph(const std::vector<std::vector<int>>& graph, const std::vector<JacobiPrediction>& predictions);

ParallelCommit compute_parallel_commit_for_vertex(int vi, bool use_cached_prediction, const JacobiPrediction& prediction,
                                                  const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                                                  const SimParams& params, const std::vector<Vec3>& x_current, const std::vector<Vec3>& xhat,
                                                  const BroadPhase& broad_phase, const PinMap* pin_map = nullptr);

// Safe-step scalar in [0, 1] for a single vertex's Newton update. Used by
// compute_parallel_commit_for_vertex; exposed for targeted unit testing.
double compute_safe_step_for_vertex(int vi, const RefMesh& ref_mesh, const SimParams& params,
                                    const std::vector<Vec3>& x, const Vec3& delta,
                                    const BroadPhase& broad_phase);

void apply_parallel_commits(const std::vector<ParallelCommit>& commits, std::vector<Vec3>& xnew);
