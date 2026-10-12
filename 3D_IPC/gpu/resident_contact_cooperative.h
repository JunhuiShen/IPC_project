#pragma once

#include "resident_contact.cuh"
#include "resident_contact_compact.cuh"
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace ipc_gpu::resident {

// One warp claims one independent queued contact. Lane zero owns geometry;
// lanes zero through eight evaluate the nine entries of one self-Hessian.
// Every other Hessian mask uses the original scalar device evaluator.
struct CooperativeExactArenaSizes {
    std::size_t words;
    std::size_t free_slots;
};

// Scratch is private to this launch and must not overlap the scalar retry
// arena, another in-flight launch, the contact records, or the queue buffers.
// Counts are uint32 words and int slots, respectively (not byte counts).
CooperativeExactArenaSizes cooperative_exact_arena_sizes(unsigned workers);

// Opt-in execution change, cached on its first query in this process:
// IPC_GPU_COOPERATIVE_GEOMETRY_CONVERSIONS=1 converts the three exact residuals
// and four exact weights in seven lanes, each with its original private arena.
// Default OFF. All exact expressions, rounded values, scalar replay, capacity
// retries, and error codes are retained. An explicit launch argument overrides
// the environment for equivalence tests without changing process configuration.
bool cooperative_geometry_conversions_enabled();

// `next` must initially be zero; `retry_count` is appended atomically and must
// have been cleared by the caller. A completed block writes only its claimed
// contact. The caller retains its original ordered per-node reduction and
// original 1600-limb retry launch. Launches asynchronously on `stream`.
// Error codes match the scalar contact pipeline: 20 invalid, 21 zero gap,
// 22 nonfinite derivatives, 23 capacity, 24 arena integrity failure.
void launch_cooperative_exact_contacts(
    const ContactInput* inputs, ContactOutput* outputs,
    const int* exact_queue, const unsigned* count,
    int* retry_queue, unsigned* retry_count, unsigned* next,
    std::uint32_t* words, int* free_slots, unsigned workers,
    int* failed, cudaStream_t stream = nullptr,
    CompactDeformableContact* compact_outputs = nullptr,
    bool parallel_geometry_conversions = cooperative_geometry_conversions_enabled());

} // namespace ipc_gpu::resident
