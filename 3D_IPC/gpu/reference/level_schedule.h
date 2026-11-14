#pragma once
#include "broadphase.h"
#include <cstddef>
#include <memory>

namespace ipc_gpu::solver::level_predictor {
struct DeviceSchedule {
    const int* vertex_colors=nullptr;
    const int* levels=nullptr;
    const int* ordered_vertices=nullptr;
    const int* group_offsets=nullptr;
    int groups=0,vertices=0;
};
// GPU-only graph/scheduling carrier. Existing GPU conservative CCD remains
// the sole motion authority; no trial or accepted position reaches the host.
// Calls and all associated CCD updates must be serialized on the default
// stream. A new prepare() invalidates every previous view/certificate.
class GpuLevelSchedule {
public:
    GpuLevelSchedule();
    ~GpuLevelSchedule();
    GpuLevelSchedule(const GpuLevelSchedule&)=delete;
    GpuLevelSchedule& operator=(const GpuLevelSchedule&)=delete;

    // Contacts MUST be candidate-only swept-box contacts, without elastic
    // dependencies. Vertices are original node IDs, never rigid-body blocks.
    // Rigid targets equal starts. Boxes enclose every allowed intermediate
    // position. Invalid geometry/indices fails closed before scheduling.
    void prepare(int vertices,const BpContact* contacts,std::size_t count,
        const double (*start)[3],const double (*targets)[3],const BpAabb* boxes,
        const unsigned char* deformable,double minimum_separation,
        bool heavy_first,int* error,bool original_projected_prune=false);

    // Run exactly one GPU legacy-colored CCD trial with this color order,
    // fixed targets/boxes, and separated() removed from BOTH incidence/query
    // inputs. Rigid vertices remain obstacles and must not be updated.
    DeviceSchedule trial()const;
    // trial_positions are the represented trial endpoints. This computes
    // stableNorm priorities and level/color groups, WITHOUT committing or
    // restoring physical state. Restore the saved start state before running
    // any actual level group. Trial completion never counts as convergence.
    DeviceSchedule finish_trial(const double (*trial_positions)[3],int* error);

    // Project the CURRENT trial/actual group IDs onto unique deformable update
    // blocks. Every rigid block is -1. The returned array replaces the old
    // broadphase block_colors for incidence, clipper and commit/worklists.
    // This never rewrites public topology and validates node/block ownership.
    const int* project_blocks(const int* node_blocks,int blocks,int* error);

    // Feed ONLY these compacted contacts to both incidence and CCD. Order is
    // stable relative to the original candidates. This prevents a second
    // pruning implementation from keeping reads omitted by this graph.
    const BpContact* retained_contacts()const;
    std::size_t retained_contact_count()const;
    // Original-candidate index space ONLY; diagnostic/graph provenance. Never
    // apply this stencil to the renumbered compacted contact array.
    const unsigned char* separated()const;
    const double* movement()const;
    const double (*starts()const)[3];
    const double (*targets()const)[3];
    const BpAabb* boxes()const;
    const std::size_t* graph_offsets()const;
    const int* graph_neighbors()const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace ipc_gpu::solver::level_predictor
