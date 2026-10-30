#pragma once
#include "scene.h"
#include "contact_worklists.h"
#include <memory>

namespace ipc_gpu::solver {

// These codes are written atomically to the caller's device error word. No
// failure authorizes an unverified update. The enclosing solver must not
// commit if any stage reports an error.
enum class LinearClipError : int {
    InvalidInput=31, InitialContact=32, ExactResources=33,
    Unresolved=34, OutsideBlueBox=35, UnsupportedMotion=36
};

// GPU-only proposed-step validation/backtracking. The CSR maps update blocks
// to unique indices into BroadPhaseView.contacts. The broad-phase boxes must
// enclose the current positions and every proposed/accepted linear path.
// Accepted endpoints are written once and must be copied, not reconstructed.
// Host work is allocation and kernel dispatch only; no geometry is read back.
// Initial deformable endpoint predicates run one GPU worker per contact;
// proven-safe predicates are reused only for that unchanged initial proposal.
// Backtracking always invalidates this per-color certificate cache.
// This initial exact cold queue is serialized, using one reusable global
// integer arena. Exact kernels use a translation-unit device pool context:
// all instances must be serialized on the same CUDA stream (the GPU
// solver uses the default stream). Concurrent exact calls are not supported.
//
// ORIGINAL modes (OriginalSolver/OriginalColoredGuess) run a BOUNDED parallel
// whole-node resolve: each of a small fixed pool of workers owns a disjoint
// reusable exact arena slice and processes a stride of the active color's
// nodes. Workers only read the immutable current positions, and the complete
// kernel join precedes any commit or the next color.
class GpuLinearClipper {
public:
    GpuLinearClipper();
    ~GpuLinearClipper();
    GpuLinearClipper(const GpuLinearClipper&)=delete;
    GpuLinearClipper& operator=(const GpuLinearClipper&)=delete;

    // Deformable proposals are first clamped COORDINATEWISE to each node's
    // blue box with the CPU's fixed 1e-10 inset, and the stored endpoint is
    // before + (clipped - before), before any contact/CCD validation. This is
    // per-vertex; rigid translation keeps its shared sweep fraction.
    //
    // `original_cooperative_selected` is the solver-computed device node mask
    // for the CPU ColoredContactSweep family. It is consulted only when the
    // CPU selected-vertex squared cutoff applies (original modes, and robust
    // exact mode with CCD disabled); predictors pass StoredNoop and no mask.
    // Every device mask value must be 0 or 1; invalid masks fail closed before
    // any publish, and are validated once per call.
    void clip_deformable_color(const DeformableScene&,const DeformableState&,
        const ElasticWorkspace&,const BroadPhaseView&,
        const std::size_t* block_contact_offsets,const int* contact_indices,
        const BpAabb* blue_boxes,const int* block_colors,int color,
        double (*accepted_positions)[3],int* error,bool use_ccd=true,cudaStream_t stream=nullptr,
        ColorContactWorklist color_pairs={},bool exact_step_bound=false,
        LinearClipMode mode=LinearClipMode::Robust,
        OriginalUpdateConvention convention=OriginalUpdateConvention::StoredNoop,
        const unsigned char* original_cooperative_selected=nullptr);

    void clip_rigid_translation_color(const RigidScene&,const RigidState&,
        const RigidWorkspace&,const DeformableState&,const BroadPhaseView&,
        const std::size_t* block_contact_offsets,const int* contact_indices,
        const BpAabb* blue_boxes,const int* block_colors,int color,
        double (*accepted_proxy_positions)[3],int* error,bool use_ccd=true,cudaStream_t stream=nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ipc_gpu::solver
