#pragma once

#include "broadphase.h"
#include "dat_types.h"
#include <cuda_runtime_api.h>
#include <cstddef>
#include <memory>

namespace ipc_gpu::solver {

// Device implementation of CPU planar_DAT. It filters simultaneous linear
// increments, not absolute endpoints. It has no barrier/SDF forces, CCD,
// tetrahedron-inversion check, rigid-body filter or CPU geometry fallback.
//
// All numerical pointers are device pointers. prepare caps increments and
// returns blue boxes centered at the unchanged positions with half-extent
// query_radius*padding. The caller must check device status, then build fresh
// CandidatesOnly contacts with d_hat=0 and protect_linear_gap=false. Its
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
// copied to the host. Exact EE failures retry with a larger device arena.
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
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ipc_gpu::solver
