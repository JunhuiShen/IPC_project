#pragma once

#include <cstddef>
#include <memory>

namespace ipc_gpu {
namespace resident {

// Plain, device-resident geometry. Coordinates and bounds are doubles.
struct BpAabb { double lo[3], hi[3]; };
struct BpTriangle { int vertices[3]; };
struct BpEdge { int vertices[2]; };
struct BpDependency { int vertices[4]; int count; };
struct BpContact {
    int vertices[4];
    // 0: node/triangle [point,a,b,c], 1: edge/edge [a,b,c,d].
    int kind;
};

struct BroadPhaseTopology {
    int vertex_count = 0;
    int block_count = 0;
    int triangle_count = 0;
    int edge_count = 0;
    int dependency_count = 0;
    const BpTriangle* triangles = nullptr;
    const BpEdge* edges = nullptr;
    // Additional elastic cliques: four hinge vertices and four tet vertices.
    // Triangle dependencies are inserted automatically.
    const BpDependency* dependencies = nullptr;
    // Each deformable node is its own update block. All vertices of a rigid
    // body map to one block, so conflicts cannot separate the body's vertices.
    const int* vertex_block = nullptr;
    // Null means every vertex issues NT queries; otherwise 0 skips interiors.
    const unsigned char* surface_query = nullptr;
    // Initial-guess CCD sweeps use contact-only coloring. Set false there
    // and omit explicit elastic dependencies; normal solver default is true.
    bool include_triangle_dependencies = true;
    // Preserve the linear CCD numerical gap even if it exceeds d_hat. Query
    // padding is enlarged on-device from the complete allowed blue regions;
    // the blue movement limits and barrier activation distance are unchanged.
    bool protect_linear_gap = false;
};

struct BroadPhaseView {
    const BpContact* contacts = nullptr;
    std::size_t contact_count = 0;
    const std::size_t* adjacency_offsets = nullptr;
    const int* adjacency = nullptr;
    std::size_t directed_edge_count = 0;
    const int* block_colors = nullptr;
    const int* color_blocks = nullptr;
    const int* color_offsets = nullptr;
    int color_count = 0;
};

// Candidate-only builds retain the identical validation, BVH traversal and
// ordered contact array. Graph/color fields in their returned view are empty,
// including after reusing an owner that previously built a solver graph.
enum class BroadPhaseMode { Solver, CandidatesOnly };

// Work-presence bits, not collision filters. Fixed rigid bodies contribute no
// bits; pinned deformable nodes still contribute DeformableColorStage because
// their pins are soft energies. Candidate lists and coloring are unchanged.
enum ColorStage : unsigned int {
    DeformableColorStage = 1u,
    RigidTranslationColorStage = 2u,
    RigidOrientationColorStage = 4u
};

// OR device-resident block stage bits into one mask per existing color, on the
// default stream. Inputs must belong to the current valid broadphase view.
// The caller owns/reuses both buffers (block_count and color_count entries).
// Only these orchestration masks may be copied to the host to omit empty
// kernel stages; no evolving positions, contacts, or derivatives are read back.
void collect_color_stage_masks(const BroadPhaseView& view, int block_count,
                              const unsigned int* block_stages,
                              unsigned int* color_stages);

// Creates conservative endpoint-motion boxes entirely on the GPU. Per axis,
// expand the [start,end] extent by padding_fraction * extent / 2 on each side
// (0.2 therefore means 120% of the displacement extent), plus at least
// max(absolute_pad, 1e-8, 128*epsilon*scene_scale). scene_scale is the largest
// absolute coordinate across BOTH start and target arrays, reduced on-device.
// This numerical-gap padding is needed even for zero d_hat: it retains pairs
// whose paths cannot intersect but whose endpoint gap must be protected.
// This is valid for linear vertex motion, NOT a bound for a rotational arc.
// Arrays contain 3 * count doubles. Outboxes are device memory.
void make_motion_boxes(const double* start, const double* end, int count,
                       double padding_fraction, double absolute_pad,
                       BpAabb* outboxes);

// Every input/output pointer is a CUDA device pointer. The host only schedules
// kernels, allocates memory, and reads counts/status; geometry, candidate lists,
// graph construction and coloring never fall back to CPU computation.
//
// Input blue boxes MUST enclose every position allowed until the next rebuild.
// The solver must clip motion to those boxes (rigid rotations require a bound
// on the entire arc). Rebuild after changing that permitted region. d_hat is
// applied conservatively to contact queries, not to the blue motion bounds.
//
// Baseline implementation: Morton-sorted balanced BVHs, uncapped two-pass
// contact compaction, sorted unique dependency CSR, deterministic serial GPU
// greedy coloring. The latter is correctness-first, not a performance claim.
class ResidentBroadPhase {
public:
    ResidentBroadPhase();
    ~ResidentBroadPhase();
    ResidentBroadPhase(const ResidentBroadPhase&) = delete;
    ResidentBroadPhase& operator=(const ResidentBroadPhase&) = delete;

    BroadPhaseView rebuild(const BroadPhaseTopology& topology,
                           const BpAabb* blue_boxes, double d_hat,
                           BroadPhaseMode mode = BroadPhaseMode::Solver);
    BroadPhaseView view() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace resident
} // namespace ipc_gpu
