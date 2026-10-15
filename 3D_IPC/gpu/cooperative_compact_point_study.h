#pragma once
// Isolated study API. No frozen production dispatcher calls this function.
#include "resident_contact_cooperative.h"

namespace ipc_gpu::resident {
// Hot inplace-ratio mode only. Exactly32 threads/block, one query owner and
// nine entry lanes; workers reuses min(candidate_count,256), not a new policy.
// The caller must provide the EXISTING cooperative arena shape returned by
// cooperative_exact_arena_sizes(workers), never the scalar512-slot layout.
// Queues/counts/cursor and output contracts match the frozen C8 first tier.
// Owner/helper pools use512/128 slots with256-word stride; C8 is arithmetic
// capacity only. Next and fallback_count initially zero; unique valid IDs;
// both queue capacities at least candidate_count. No buffer aliases.
//
// After this launch completes in the SAME stream, unresolved B may feed the
// unchanged v6 cooperative kernel, reusing this arena only at that ordered
// kernel boundary. It then feeds the original1600 stage. No Integer/view or
// scratch ownership survives a launch. No inline256 evaluator is present.
// Do not stage/compile/execute until the scalar C8 candidate is qualified and
// timed. This prototype is not a production/default change or speed claim.
void launch_cooperative_compact_point_inplace_study(
    const ContactInput* inputs,ContactOutput* outputs,
    const int* queue,const unsigned* count,
    int* fallback_queue,unsigned* fallback_count,unsigned* next,
    std::uint32_t* cooperative_words,int* cooperative_slots,unsigned workers,
    int* failed,cudaStream_t stream,
    CompactDeformableContact* compact_outputs,unsigned char* output_activity);
} // namespace ipc_gpu::resident
