#pragma once
#include "resident_contact.cuh"
#include "resident_contact_compact.cuh"
#include <cuda_runtime_api.h>
#include <cstdint>

namespace ipc_gpu::resident {
// A default-off pipeline first tier. Each scalar worker has the ORIGINAL
// 256-word/512-slot arena geometry; C8 is an arithmetic bound, not a new stride.
// Reserve at least min(candidate_count,workers)*256*512 words and the matching
// min(candidate_count,workers)*512 free-list entries. Input queue identities
// are unique and valid, *count<=candidate_count; both queues have that capacity.
// `next` and *fallback_count start at zero. Buffers do not overlap. The pipeline
// retains its original cooperative arena reservation/fresh-initialization
// geometry, so the skipped original512/128-slot cooperative path has proved headroom.
//
// Success publishes the original full/compact/activity contract. Every
// unsupported or unsuccessfully evaluated pair appends to fallback_queue only
// after complete owner destruction and pool accounting. Leaks/corruption set
// the original24 latch. No scalar256/1600 evaluator is inlined into this TU.
// Caller must run the unchanged cooperative256 and1600 stages on unresolved
// pairs, ordered on the same stream, before any reduction or position commit.
// Zero work returns before pool access. workers/threads reuse the original
// small-worker schedule; threads is a power of two in[1,32], dividing workers.
void launch_compact_point_contacts(
    const ContactInput* inputs,ContactOutput* outputs,
    const int* queue,const unsigned* count,
    int* fallback_queue,unsigned* fallback_count,unsigned* next,
    std::uint32_t* words,int* free_slots,unsigned workers,unsigned threads,
    int* failed,cudaStream_t stream,
    CompactDeformableContact* compact_outputs,unsigned char* output_activity,
    bool inplace_ratio);
} // namespace ipc_gpu::resident
