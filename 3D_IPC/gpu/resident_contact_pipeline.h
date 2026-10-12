#pragma once
#include "resident_scene.h"
#include "resident_contact_worklists.h"
#include <memory>

namespace ipc_gpu::resident {

enum class ContactAssemblyPhase { Deformable, RigidTranslation, RigidOrientation };
struct ContactIncidence {
    const std::size_t* block_offsets=nullptr;
    const int* pair_indices=nullptr;
    ColorContactWorklist color_pairs{};
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
};

// GPU incident lists, gather, ordinary and exact contact evaluation, ordered
// accumulation, and analytic SDF assembly. No derivative/geometry array is
// copied to or computed on the CPU. Exact queries use independent per-worker
// scratch arenas; a second kernel retries arithmetic-capacity failures with
// the original larger capacity. Ordered per-block accumulation is unchanged.
// This pipeline uses the default CUDA stream; calls on the same pipeline must
// remain ordered because its workspace is reused. The enclosing solver owns
// state rollback after an error.
class ResidentContactPipeline {
public:
    ResidentContactPipeline();
    ~ResidentContactPipeline();
    ResidentContactPipeline(const ResidentContactPipeline&)=delete;
    ResidentContactPipeline& operator=(const ResidentContactPipeline&)=delete;

    // Call after every broad-phase rebuild. vertex_block and surface_query are
    // device arrays from that topology. A null surface_query means all nodes
    // are surface nodes; otherwise solid-interior SDF updates are excluded.
    void rebuild(const BroadPhaseView&,int block_count,const int* vertex_block,
                 const unsigned char* surface_query=nullptr,
                 const ContactBoxPruning& pruning={});
    ContactIncidence incidence()const;
    void assemble_color(const DeformableScene&,const DeformableState&,
                        const ElasticWorkspace&,const RigidScene&,const RigidState&,
                        const RigidWorkspace&,const SdfScene&,
                        const ContactParameters&,int color,ContactAssemblyPhase);

    // ElasticWorkspace.error is preserved if already nonzero. New codes:
    // 20 invalid contact input, 21 exact zero gap, 22 nonfinite contact,
    // 23 exact bit capacity, 24 exact arena failure, 25 SDF failure,
    // 26 nonfinite accumulated system, 27 invalid deformable block mapping.
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ipc_gpu::resident
