#pragma once
#include "scene.h"
#include "contact_worklists.h"
#include "rigid_pins_types.h"
#include <memory>

namespace ipc_gpu::solver {

enum class ContactAssemblyPhase { Deformable, RigidTranslation, RigidOrientation };
enum class ContactAssemblyMode { ColorSweep, FrozenDeformableSnapshot };
struct RigidPinAssembly {
    RigidPinScene scene{};
    double stiffness=0.0, time=0.0;
};
struct ContactIncidence {
    const std::size_t* block_offsets=nullptr;
    const int* pair_indices=nullptr;
    ColorContactWorklist color_pairs{};
    // Original CPU groups are ordered before cache retirement. Null means
    // the retained incidence is also the raw incidence for this rebuild.
    const std::size_t* raw_block_counts=nullptr;
};
// Optional rebuild-lifetime certificate inputs. Only contacts whose four
// vertices are deformable are eligible; rigid rotational arcs are not covered
// by a certificate on their endpoint boxes.
struct ContactBoxPruning {
    const BpAabb* boxes=nullptr;
    const unsigned char* deformable=nullptr;
    const double (*positions)[3]=nullptr;
    const double (*previous)[3]=nullptr;
    ContactParameters parameters{};
    // CPU original-mode certificate, requested by resolved v2 dispatch.
    // Kept separate from the optional robust execution certificate.
    bool original_geometry=false;
};

// GPU incident lists, gather, ordinary and exact contact evaluation, ordered
// accumulation, and analytic SDF assembly. No derivative/geometry array is
// copied to or computed on the CPU. Exact queries use independent per-worker
// scratch arenas; a second kernel retries arithmetic-capacity failures with
// the original larger capacity. Ordered per-block accumulation is unchanged.
// This pipeline uses the default CUDA stream; calls on the same pipeline must
// remain ordered because its workspace is reused. The enclosing solver owns
// state rollback after an error.
class GpuContactPipeline {
public:
    GpuContactPipeline();
    ~GpuContactPipeline();
    GpuContactPipeline(const GpuContactPipeline&)=delete;
    GpuContactPipeline& operator=(const GpuContactPipeline&)=delete;

    // Call after every broad-phase rebuild. vertex_block and surface_query are
    // device arrays from that topology. A null surface_query means all nodes
    // are surface nodes; otherwise solid-interior SDF updates are excluded.
    void rebuild(const BroadPhaseView&,int block_count,const int* vertex_block,
                 const unsigned char* surface_query=nullptr,
                 const ContactBoxPruning& pruning={});
    ContactIncidence incidence()const;
    // Explicit current-color launch bound from this completed rebuild. When
    // IPC_GPU_CCD_CURRENT_COLOR_LAUNCH is off, or indexed queues are absent,
    // retain incidence()'s original bound. No per-color device readback.
    // The view must be consumed for this color before the next rebuild.
    ContactIncidence incidence(int color)const;
    void assemble_color(const DeformableScene&,const DeformableState&,
                        const ElasticWorkspace&,const RigidScene&,const RigidState&,
                        const RigidWorkspace&,const SdfScene&,
                        const ContactParameters&,int color,ContactAssemblyPhase,
                        ContactAssemblyMode mode=ContactAssemblyMode::ColorSweep,
                        const RigidPinAssembly& pins={},
                        ContactGeometryPolicy geometry=ContactGeometryPolicy::Robust);

    // ElasticWorkspace.error is preserved if already nonzero. New codes:
    // 20 invalid contact input, 21 exact zero gap, 22 nonfinite contact,
    // 23 exact bit capacity, 24 exact arena failure, 25 SDF failure,
    // 26 nonfinite accumulated system, 27 invalid deformable block mapping.
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ipc_gpu::solver
