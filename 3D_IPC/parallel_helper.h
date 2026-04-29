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

struct RedBoxes {
    std::vector<AABB> tri;
    std::vector<AABB> edge;
};

struct GreenBoxes {
    std::vector<AABB> tri;
    std::vector<AABB> edge;
};

// Hot-path Jacobi stage used by the basic parallel solver: compute only g/delta.
void build_jacobi_prediction_deltas(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                                    const SimParams& params, const std::vector<Vec3>& positions, const std::vector<Vec3>& xhat,
                                    const BroadPhase::Cache& bp_cache, std::vector<JacobiPrediction>& predictions,
                                    const PinMap* pin_map = nullptr);

// Build isotropic blue boxes from prediction deltas and mirror them into
// prediction.certified_region; optionally also writes them into blue_boxes_out.
// If provided, per_vertex_radii overrides ||delta|| for each vertex.
void build_blue_boxes(const std::vector<Vec3>& positions,
                      bool use_parallel,
                      std::vector<JacobiPrediction>& jacobi_predictions,
                      std::vector<AABB>* blue_boxes_out = nullptr,
                      const std::vector<double>* per_vertex_radii = nullptr);

// Build red triangle/edge boxes as unions of incident blue boxes.
void build_red_boxes(const RefMesh& ref_mesh,
                     const std::vector<std::array<int, 2>>& edges,
                     const std::vector<AABB>& blue_boxes,
                     RedBoxes& red_boxes);

// Build green triangle/edge boxes by padding red boxes by d_hat.
void build_green_boxes(const RedBoxes& red_boxes, double d_hat, GreenBoxes& green_boxes);

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
std::vector<std::vector<int>> build_elastic_adj(const RefMesh& ref_mesh, const VertexTriangleMap& adj, int num_vertices);

// Contact edges of the conflict graph from broad-phase NT/SS pair lists.
// Stable between broad-phase initialize() calls (key on BroadPhase::version()).
std::vector<std::vector<int>> build_contact_adj(const BroadPhase::Cache& bp_cache, int num_vertices);

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

// Infinity norm of mass-normalized predicted gradients.
double compute_prediction_residual_inf_norm(const RefMesh& ref_mesh,
                                            const std::vector<JacobiPrediction>& predictions,
                                            bool use_parallel);

void compute_local_newton_direction(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                                    const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                                    const BroadPhase::Cache& bp_cache, Vec3& g_out, Mat33& H_out, Vec3& delta_out,
                                    const PinMap* pin_map = nullptr,
                                    const std::vector<TriPrecompute>* tri_cache = nullptr,
                                    const std::vector<HingePrecompute>* hinge_cache = nullptr);

double clip_step_to_certified_region(int vi, const std::vector<Vec3>& x, const Vec3& fresh_delta, const AABB& certified_region);

double compute_safe_step_for_vertex(int vi, const RefMesh& ref_mesh, const SimParams& params,
                                    const std::vector<Vec3>& x, const Vec3& delta,
                                    const BroadPhase::Cache& bp_cache);

// Register pairs using already-constructed green boxes.
BroadPhase::Cache register_barrier_pairs_from_blue_and_green(
    const RefMesh& ref_mesh,
    const std::vector<std::array<int, 2>>& edges,
    const std::vector<AABB>& blue_boxes,
    const GreenBoxes& green_boxes);
