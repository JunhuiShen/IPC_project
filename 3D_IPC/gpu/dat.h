#pragma once

#include "broadphase.h"
#include "dat_types.h"
#include "scene_types.h"
#include <cuda_runtime_api.h>
#include <cstddef>
#include <memory>

namespace ipc_gpu::solver {

// Device DAT filtering. The linear overload pair implements CPU planar_DAT
// for simultaneous increments. The mixed overload pair also certifies curved
// rigid trajectories. This component has no barrier/SDF forces, CCD,
// tetrahedron-inversion check or CPU geometry fallback.
//
// Linear overload contract: all numerical pointers are device pointers.
// prepare caps increments and
// returns blue boxes centered at the unchanged positions with half-extent
// query_radius*padding. The caller must check device status, then build fresh
// CandidatesOnly contacts with d_hat=0 and protect_linear_gap=false. An
// existing same-snapshot conservative superset may be shared with force
// assembly instead of rebuilding. Its
// topology must exclude incident primitives and solid-interior NT queries.
// filter computes per-pair planes from those same positions/capped updates,
// takes the minimum factor per vertex, and produces accepted INCREMENTS.
// Neither operation modifies the input positions or raw increments. The
// accepted output must not overlap either input array.
//
// The caller must validate x+accepted before atomically publishing a complete
// Jacobi sweep. Error status must be checked before publishing. Calls enqueue
// work on the supplied stream; first/growing allocations may synchronize.
// Keep positions, raw updates and candidate view alive and unchanged until
// their consumers finish. Each prepared view is consumed once, on its stream.
// Counts/status are orchestration only; no numerical arrays/queue counts are
// copied to the host. Uncertain NT/EE geometry retries in device exact arenas.
class GpuDatFilter {
public:
    GpuDatFilter();
    ~GpuDatFilter();
    GpuDatFilter(const GpuDatFilter&) = delete;
    GpuDatFilter& operator=(const GpuDatFilter&) = delete;
    const BpAabb* prepare(std::size_t count, const double (*positions)[3],
        const double (*raw_increments)[3], const DatParameters&, int* error,
        cudaStream_t stream = nullptr);
    void filter(const BroadPhaseView&, double (*accepted_increments)[3],
        int* error, cudaStream_t stream = nullptr);

    // Mixed Jacobi DAT. Every pointer is device storage and all inputs remain
    // immutable. RigidState supplies current com/omega and previous_orientation
    // (the physical substep's q_n). The proposed generalized increments share
    // one accepted factor per body. Curved trajectories are certified over the
    // full prefix, including their displacement from represented query nodes.
    // Fixed translate/rotate modes require zero corresponding increments.
    const BpAabb* prepare(std::size_t count, const double (*positions)[3],
        const double (*raw_vertex_increments)[3], const RigidScene&,
        const RigidState&, const double (*delta_com)[3],
        const double (*delta_omega)[3], double dt, const DatParameters&,
        int* error, cudaStream_t stream = nullptr);
    // These are accepted ABSOLUTE positions/poses, already checked as actually
    // represented. They must be separate from every input and from each other.
    // Publish all four arrays together only after checking error. Candidate
    // topology must use one vertex_block per rigid body and independent blocks
    // for deformable vertices, and exclude solid-interior NT queries.
    void filter(const BroadPhaseView&, double (*accepted_positions)[3],
        double (*accepted_com)[3], double (*accepted_omega)[3],
        double (*accepted_orientation)[4], int* error,
        cudaStream_t stream = nullptr);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ipc_gpu::solver
