#pragma once

#include "kernels.h"
#include "solid_types.h"
#include <cstddef>
#include <cuda_runtime_api.h>
#include <memory>

namespace ipc_gpu::solver {

// All pointers in these views are DEVICE pointers. Host code may allocate,
// upload static topology/materials, and enqueue kernels, but never gathers
// element positions or accumulates numerical results on the CPU.
struct TriangleIncident { int owner, nodes[3]; Triangle material; };
struct SolidIncident { int owner, nodes[4]; Solid material; };
struct HingeIncident { int owner, nodes[4]; Hinge material; };

struct DeformableScene {
    std::size_t nodes, triangle_count, solid_count, hinge_count;
    const int* node_blocks;
    const unsigned char* deformable;
    const unsigned char* solid;
    const unsigned char* pinned;
    const double* masses;
    const double (*pin_targets)[3];
    const TriangleIncident* triangles;
    const SolidIncident* solids;
    const HingeIncident* hinges;
    const std::size_t *triangle_offsets, *solid_offsets, *hinge_offsets;
};

struct DeformableState {
    double (*positions)[3];
    double (*velocities)[3];
    double (*previous)[3];
    double (*predicted)[3];
};

struct ElasticWorkspace {
    Point* points;
    Triangle* triangles;
    Solid* solids;
    Hinge* hinges;
    Derivatives *point_outputs, *triangle_outputs, *solid_outputs, *hinge_outputs;
    Derivatives* systems;
    double (*proposed)[3];
    // Device status: 0 success, 1 nonfinite derivative, 2 invalid/singular
    // node solve. The enclosing solver owns clear/check/rollback policy.
    int* error;
};

// A stable scheduling view, never a compacted numerical output array. Every
// index addresses the ORIGINAL scene/workspace storage. Per-node reductions
// still traverse the original incident CSR ranges in their original order.
struct ElasticIndexRange {
    const std::size_t* indices=nullptr;
    std::size_t count=0;
};
struct ElasticColorWorklist {
    bool enabled=false;
    ElasticIndexRange nodes,triangles,solids,hinges;
    unsigned node_threads=1,element_threads=8;
    const int* block_colors=nullptr;
    int color=-1;
};

// Opt in with IPC_GPU_ELASTIC_COLOR_WORKLISTS=1 (default0). Rebuild after
// EVERY coloring or owner/topology change, before obtaining any color view.
// Stable GPU sorting preserves original array-index order within each color.
// Only offsets/counts and validation status return to the host; geometry,
// gathered materials, and computed derivatives never do. Rebuild synchronizes
// its supplied stream to return launch metadata; assembly remains asynchronous.
// Old views are invalid after rebuild. The caller must finish consumers of
// old views before rebuilding, or order them on this same stream. Disabled
// instances enqueue no work.
// Optional launch widths IPC_GPU_ELASTIC_NODE_THREADS (default1) and
// IPC_GPU_ELASTIC_ELEMENT_THREADS (default8) accept powers of two through128.
class GpuElasticWorklists {
public:
    GpuElasticWorklists();
    ~GpuElasticWorklists();
    GpuElasticWorklists(const GpuElasticWorklists&)=delete;
    GpuElasticWorklists& operator=(const GpuElasticWorklists&)=delete;
    bool enabled()const;
    void rebuild(const DeformableScene&,const int* block_colors,int block_count,
                 int color_count,cudaStream_t stream=nullptr);
    ElasticColorWorklist color(int color)const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Each operation only enqueues GPU work. None copies numerical arrays to host,
// calls a CPU solver, or synchronizes the device. Stream order is the phase
// boundary. Contact/SDF assembly adds into systems after elastic assembly.
void begin_deformable_substep(const DeformableScene&, const DeformableState&,
                             double dt, cudaStream_t stream = nullptr);
void assemble_elastic_color(const DeformableScene&, const DeformableState&,
                            const ElasticWorkspace&, const int* block_colors,
                            int color, const Parameters&, const SolidParameters&,
                            cudaStream_t stream = nullptr,ElasticColorWorklist worklist = {});
void propose_deformable_color(const DeformableScene&, const DeformableState&,
                             const ElasticWorkspace&, const int* block_colors,
                             int color, double damping, cudaStream_t stream = nullptr,
                             ElasticColorWorklist worklist = {});
// Frozen Jacobi proposals are displacements, not rounded endpoints. In
// particular, zero damping must skip the inverse and large coordinates must
// not erase a small displacement before DAT has filtered it.
void propose_deformable_displacements(const DeformableScene&,
                                     const ElasticWorkspace&, double damping,
                                     double (*increments)[3],
                                     cudaStream_t stream = nullptr);
// Fractions are device CCD/box-clip outputs. This entry point is not a CCD
// implementation and is deliberately not exposed as a complete solver.
void commit_deformable_color(const DeformableScene&, const DeformableState&,
                            const ElasticWorkspace&, const int* block_colors,
                            int color, const double* certified_fractions,
                            cudaStream_t stream = nullptr,ElasticColorWorklist worklist = {});
// Preferred CCD integration: copy the already-certified binary64 endpoint.
void commit_deformable_endpoints(const DeformableScene&, const DeformableState&,
                                const ElasticWorkspace&, const int* block_colors,
                                int color, const double (*accepted)[3],
                                cudaStream_t stream = nullptr,ElasticColorWorklist worklist = {});
void finish_deformable_substep(const DeformableScene&, const DeformableState&,
                              double dt, cudaStream_t stream = nullptr);

} // namespace ipc_gpu::solver
