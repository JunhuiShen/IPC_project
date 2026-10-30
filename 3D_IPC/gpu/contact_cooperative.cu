#include "contact_cooperative.h"
#include "contact_cooperative.cuh"
#include "point_hessian_canonical.cuh"
#include "contact_execution.h"
#include <cuda_runtime.h>
#include <limits>
#include <new>
#include <stdexcept>

namespace ipc_gpu::solver {
namespace {
namespace coop = contact_cooperative_detail;
namespace exact = ipc_gpu::exact;

struct BlockState {
    coop::SharedGeometry geometry;
    coop::SharedConversions conversions;
    ContactOutput* output;
    int pair;
    int role;
    unsigned work;
    unsigned prepared;
    unsigned replay;
    unsigned corrupt;
};

template<bool InplaceRatio> __device__ __forceinline__ bool prepare(const ContactInput& in, int role,
    coop::GeometryOwner<coop::capacity>& owner, BlockState& state) {
    return coop::prepare_geometry<coop::capacity,InplaceRatio>(in, role, owner, *state.output, state.geometry);
}
template<bool InplaceRatio> __device__ __forceinline__ coop::EntryResult entry(const coop::SharedGeometry& geometry, int index) {
    return coop::compute_entry<coop::capacity,InplaceRatio>(geometry, index);
}
__device__ __forceinline__ void finish(const ContactInput& in,
    const coop::GeometryOwner<coop::capacity>& owner, ContactOutput& output) {
    coop::finish_output(in, owner, output);
}
__device__ __forceinline__ ContactOutput scalar(const ContactInput& in) {
    return evaluate_contact_exact<coop::capacity>(in);
}
__device__ int status_code(ContactStatus status) {
    if (status == ContactStatus::InvalidInput) return 20;
    if (status == ContactStatus::ZeroDistance) return 21;
    if (status == ContactStatus::ExactCapacityExceeded) return 23;
    return 22;
}

template<bool ParallelConversions,bool InplaceRatio>
__global__ void cooperative_contacts(
    const ContactInput* inputs, ContactOutput* outputs,
    const int* queue, const unsigned* count,
    int* retry_queue, unsigned* retry_count, unsigned* next,
    std::uint32_t* words, int* free_slots, int* failed,
    CompactDeformableContact* compact_outputs, unsigned char* output_activity) {
    const unsigned lane = threadIdx.x;
    const unsigned worker = blockIdx.x;
    // A zero-work launch must not touch an arena slice that was not reserved.
    if (worker >= *count) return;
    __shared__ BlockState state;
    __shared__ __align__(16) unsigned char output_storage[sizeof(ContactOutput)];
    if (lane == 0) state.output = new (output_storage) ContactOutput{};
    exact::DevicePool pool;
    const int private_slots = lane == 0 ? coop::owner_slots : coop::helper_slots;
    const std::size_t offset = std::size_t(worker)*coop::slots_per_worker
        + (lane == 0 ? 0 : coop::owner_slots+(lane-1)*coop::helper_slots);
    if (lane < coop::active_lanes)
        exact::initialize_pool(pool, words+offset*coop::capacity,
            free_slots+offset, coop::capacity, private_slots);
    using Owner = coop::GeometryOwner<coop::capacity>;
    alignas(Owner) unsigned char owner_storage[sizeof(Owner)];
    Owner* owner = nullptr;
    __syncthreads();
    for (;;) {
        if (lane == 0) {
            state.work = 0; state.prepared = 0; state.replay = 0; state.corrupt = 0;
            if (!atomicAdd(failed, 0)) {
                const unsigned index = atomicAdd(next, 1u);
                if (index < *count) {
                    state.work = 1; state.pair = queue[index];
                    state.role = coop::self_block(inputs[state.pair].hessian_blocks);
                    state.replay = state.role < 0;
                }
            }
        }
        __syncthreads();
        if (!state.work) break;
        if (lane == 0 && !state.replay) {
            owner = new (owner_storage) Owner();
            if constexpr (ParallelConversions) {
                state.prepared = coop::prepare_feature(inputs[state.pair], *owner, *state.output);
                if (state.prepared) coop::publish_conversions(*owner, state.conversions);
            } else {
                state.prepared = prepare<InplaceRatio>(inputs[state.pair], state.role, *owner, state);
            }
            if (pool.failed || state.output->status != ContactStatus::Ok) {
                state.replay = 1; state.prepared = 0;
            }
        }
        // Publishes every immutable limb before any helper borrows it.
        __syncthreads();
        if constexpr (ParallelConversions) {
            if (state.prepared && lane < 7) {
                const auto result = coop::convert_geometry_component<coop::capacity,InplaceRatio>(state.conversions, int(lane));
                state.conversions.values[lane] = result.value;
                if (!result.ok || pool.failed) atomicOr(&state.replay, 1u);
            }
            // All borrowed adapters and conversion temporaries have died before
            // lane zero resumes preparation or eventually destroys the owner.
            __syncthreads();
            if (lane == 0 && state.prepared) {
                if (state.replay) state.prepared = 0;
                else {
                    for (int k = 0; k < 3; ++k) owner->residual[k] = state.conversions.values[k];
                    for (int i = 0; i < 4; ++i) state.output->weights[i] = state.conversions.values[3+i];
                    state.prepared = coop::prepare_derivatives(inputs[state.pair], state.role,
                        *owner, *state.output, state.geometry);
                    if (pool.failed || state.output->status != ContactStatus::Ok) {
                        state.replay = 1; state.prepared = 0;
                    }
                }
            }
            __syncthreads();
        }
        if (state.prepared && lane < coop::active_lanes) {
            const auto result = entry<InplaceRatio>(state.geometry, int(lane));
            state.output->barrier_hessian[state.role][state.role][lane] = result.value;
            if (!result.ok || pool.failed) atomicOr(&state.replay, 1u);
        }
        // Helpers have destroyed every private adapter/temporary on return.
        // No source limb may be released until this barrier has completed.
        __syncthreads();
        if (lane == 0) {
            if (state.prepared && !state.replay) {
                finish(inputs[state.pair], *owner, *state.output);
                if (state.output->status != ContactStatus::Ok) state.replay = 1;
            }
        }
        if (lane == 0 && owner) { owner->~Owner(); owner = nullptr; }
        __syncthreads();
        if (lane < coop::active_lanes) {
            if (pool.available != private_slots) atomicOr(&state.corrupt, 1u);
            if (pool.failed) {
                atomicOr(&state.replay, 1u);
                // Exhausted private helpers are not a numerical verdict. Reset
                // them only after every local Integer has been destroyed.
                exact::initialize_pool(pool, words+offset*coop::capacity,
                    free_slots+offset, coop::capacity, private_slots);
            }
        }
        __syncthreads();
        if (lane == 0) {
            if (state.corrupt) {
                atomicCAS(failed, 0, 24);
            } else {
                // Original evaluator is authoritative for unsupported masks,
                // preparation/entry failures, and capacity/error behavior.
                if (state.replay) *state.output = scalar(inputs[state.pair]);
                if (pool.failed || pool.available != pool.slot_count) {
                    atomicCAS(failed, 0, 24);
                } else if (state.output->status == ContactStatus::ExactCapacityExceeded) {
                    const unsigned retry = atomicAdd(retry_count, 1u);
                    retry_queue[retry] = state.pair;
                } else if (state.output->status != ContactStatus::Ok) {
                    atomicCAS(failed, 0, status_code(state.output->status));
                } else {
                    store_contact_output(inputs[state.pair],*state.output,outputs+state.pair,
                        compact_outputs?compact_outputs+state.pair:nullptr,
                        output_activity?output_activity+state.pair:nullptr);
                }
            }
        }
        __syncthreads();
    }
    if (lane < coop::active_lanes) exact::release_pool();
}

// Private opt-in state. The original BlockState/kernel remain byte-identical.
namespace point_canon = point_hessian_canonical;
using CanonicalPointOwner = point_canon::Owner<coop::capacity>;
struct CanonicalPointState : BlockState {
    coop::SharedGeometry canonical_geometry;
    unsigned canonical_ready;
};
__device__ __forceinline__ void prepare_canonical_point(
    const ContactInput& input, const coop::GeometryOwner<coop::capacity>& owner,
    CanonicalPointState& state, void* storage, CanonicalPointOwner*& canonical_owner) {
    if (!state.prepared || state.replay) return;
    const auto gate = point_canon::point_gate(input, state.role, owner);
    // Gate BEFORE placement construction; no canonical Integer can consume a
    // private slot when the existing owner pool cannot prove the headroom.
    if (!gate.eligible() || !point_canon::pool_bound(exact::current_pool(), coop::capacity, 256)) return;
    canonical_owner = new (storage) CanonicalPointOwner();
    const bool ready = point_canon::prepare(owner, state.role, gate,
        state.geometry, *canonical_owner, state.canonical_geometry);
    if (!ready || exact::current_pool()->failed) {
        // Keep the owner pointer for destruction after the completion barrier.
        // Never clear a failed pool or publish partial canonical views.
        state.canonical_ready = 0; state.replay = 1; state.prepared = 0;
    } else state.canonical_ready = 1;
}

template<bool ParallelConversions,bool InplaceRatio>
__global__ void canonical_point_contacts(
    const ContactInput* inputs, ContactOutput* outputs,
    const int* queue, const unsigned* count,
    int* retry_queue, unsigned* retry_count, unsigned* next,
    std::uint32_t* words, int* free_slots, int* failed,
    CompactDeformableContact* compact_outputs, unsigned char* output_activity) {
    const unsigned lane = threadIdx.x;
    const unsigned worker = blockIdx.x;
    // A zero-work launch must not touch an arena slice that was not reserved.
    if (worker >= *count) return;
    __shared__ CanonicalPointState state;
    __shared__ __align__(16) unsigned char output_storage[sizeof(ContactOutput)];
    if (lane == 0) state.output = new (output_storage) ContactOutput{};
    exact::DevicePool pool;
    const int private_slots = lane == 0 ? coop::owner_slots : coop::helper_slots;
    const std::size_t offset = std::size_t(worker)*coop::slots_per_worker
        + (lane == 0 ? 0 : coop::owner_slots+(lane-1)*coop::helper_slots);
    if (lane < coop::active_lanes)
        exact::initialize_pool(pool, words+offset*coop::capacity,
            free_slots+offset, coop::capacity, private_slots);
    using Owner = coop::GeometryOwner<coop::capacity>;
    alignas(Owner) unsigned char owner_storage[sizeof(Owner)];
    Owner* owner = nullptr;
    alignas(CanonicalPointOwner) unsigned char canonical_storage[sizeof(CanonicalPointOwner)];
    CanonicalPointOwner* canonical_owner = nullptr;
    __syncthreads();
    for (;;) {
        if (lane == 0) {
            state.work = 0; state.prepared = 0; state.replay = 0; state.corrupt = 0; state.canonical_ready = 0;
            if (!atomicAdd(failed, 0)) {
                const unsigned index = atomicAdd(next, 1u);
                if (index < *count) {
                    state.work = 1; state.pair = queue[index];
                    state.role = coop::self_block(inputs[state.pair].hessian_blocks);
                    state.replay = state.role < 0;
                }
            }
        }
        __syncthreads();
        if (!state.work) break;
        if (lane == 0 && !state.replay) {
            owner = new (owner_storage) Owner();
            if constexpr (ParallelConversions) {
                state.prepared = coop::prepare_feature(inputs[state.pair], *owner, *state.output);
                if (state.prepared) coop::publish_conversions(*owner, state.conversions);
            } else {
                state.prepared = prepare<InplaceRatio>(inputs[state.pair], state.role, *owner, state);
                prepare_canonical_point(inputs[state.pair], *owner, state,
                    canonical_storage, canonical_owner);
            }
            if (pool.failed || state.output->status != ContactStatus::Ok) {
                state.replay = 1; state.prepared = 0;
            }
        }
        // Publishes every immutable limb before any helper borrows it.
        __syncthreads();
        if constexpr (ParallelConversions) {
            if (state.prepared && lane < 7) {
                const auto result = coop::convert_geometry_component<coop::capacity,InplaceRatio>(state.conversions, int(lane));
                state.conversions.values[lane] = result.value;
                if (!result.ok || pool.failed) atomicOr(&state.replay, 1u);
            }
            // All borrowed adapters and conversion temporaries have died before
            // lane zero resumes preparation or eventually destroys the owner.
            __syncthreads();
            if (lane == 0 && state.prepared) {
                if (state.replay) state.prepared = 0;
                else {
                    for (int k = 0; k < 3; ++k) owner->residual[k] = state.conversions.values[k];
                    for (int i = 0; i < 4; ++i) state.output->weights[i] = state.conversions.values[3+i];
                    state.prepared = coop::prepare_derivatives(inputs[state.pair], state.role,
                        *owner, *state.output, state.geometry);
                    prepare_canonical_point(inputs[state.pair], *owner, state,
                        canonical_storage, canonical_owner);
                    if (pool.failed || state.output->status != ContactStatus::Ok) {
                        state.replay = 1; state.prepared = 0;
                    }
                }
            }
            __syncthreads();
        }
        if (state.prepared && lane < coop::active_lanes) {
            const auto result = point_canon::entry<coop::capacity,InplaceRatio>(
                state.geometry, state.canonical_geometry, bool(state.canonical_ready), int(lane));
            state.output->barrier_hessian[state.role][state.role][lane] = result.value;
            if (!result.ok || pool.failed) atomicOr(&state.replay, 1u);
        }
        // Helpers have destroyed every private adapter/temporary on return.
        // No source limb may be released until this barrier has completed.
        __syncthreads();
        if (lane == 0) {
            if (state.prepared && !state.replay) {
                finish(inputs[state.pair], *owner, *state.output);
                if (state.output->status != ContactStatus::Ok) state.replay = 1;
            }
        }
        if (lane == 0 && canonical_owner) {
            canonical_owner->~CanonicalPointOwner(); canonical_owner = nullptr; state.canonical_ready = 0;
        }
        if (lane == 0 && owner) { owner->~Owner(); owner = nullptr; }
        __syncthreads();
        if (lane < coop::active_lanes) {
            if (pool.available != private_slots) atomicOr(&state.corrupt, 1u);
            if (pool.failed) {
                atomicOr(&state.replay, 1u);
                // Exhausted private helpers are not a numerical verdict. Reset
                // them only after every local Integer has been destroyed.
                exact::initialize_pool(pool, words+offset*coop::capacity,
                    free_slots+offset, coop::capacity, private_slots);
            }
        }
        __syncthreads();
        if (lane == 0) {
            if (state.corrupt) {
                atomicCAS(failed, 0, 24);
            } else {
                // Original evaluator is authoritative for unsupported masks,
                // preparation/entry failures, and capacity/error behavior.
                if (state.replay) *state.output = scalar(inputs[state.pair]);
                if (pool.failed || pool.available != pool.slot_count) {
                    atomicCAS(failed, 0, 24);
                } else if (state.output->status == ContactStatus::ExactCapacityExceeded) {
                    const unsigned retry = atomicAdd(retry_count, 1u);
                    retry_queue[retry] = state.pair;
                } else if (state.output->status != ContactStatus::Ok) {
                    atomicCAS(failed, 0, status_code(state.output->status));
                } else {
                    store_contact_output(inputs[state.pair],*state.output,outputs+state.pair,
                        compact_outputs?compact_outputs+state.pair:nullptr,
                        output_activity?output_activity+state.pair:nullptr);
                }
            }
        }
        __syncthreads();
    }
    if (lane < coop::active_lanes) exact::release_pool();
}
} // namespace

CooperativeExactArenaSizes cooperative_exact_arena_sizes(unsigned workers) {
    const auto max = std::numeric_limits<std::size_t>::max();
    if (workers > max/(coop::slots_per_worker*std::size_t(coop::capacity)))
        throw std::overflow_error("cooperative exact contact arena size overflow");
    const std::size_t slots = std::size_t(workers)*coop::slots_per_worker;
    return {slots*coop::capacity, slots};
}

bool cooperative_geometry_conversions_enabled() {
    static const bool enabled = contact_execution_option("IPC_GPU_COOPERATIVE_GEOMETRY_CONVERSIONS", false);
    return enabled;
}

void launch_cooperative_exact_contacts(
    const ContactInput* inputs, ContactOutput* outputs,
    const int* exact_queue, const unsigned* count,
    int* retry_queue, unsigned* retry_count, unsigned* next,
    std::uint32_t* words, int* free_slots, unsigned workers,
    int* failed, cudaStream_t stream, CompactDeformableContact* compact_outputs,
    bool parallel_geometry_conversions, unsigned char* output_activity) {
    if (!workers) return;
    if (!inputs || !outputs || !exact_queue || !count || !retry_queue
        || !retry_count || !next || !words || !free_slots || !failed)
        throw std::invalid_argument("cooperative exact contact launch requires all buffers");
    static const bool inplace_ratio=contact_execution_option("IPC_GPU_CONTACT_INPLACE_RATIO",false);
    static const bool canonical_point = contact_execution_option("IPC_GPU_CANONICAL_POINT_HESSIAN", false);
    if (canonical_point) {
        if (parallel_geometry_conversions) {
            if (inplace_ratio)
                canonical_point_contacts<true,true><<<workers, 32, 0, stream>>>(inputs, outputs,
                    exact_queue, count, retry_queue, retry_count, next, words, free_slots, failed,compact_outputs,output_activity);
            else
                canonical_point_contacts<true,false><<<workers, 32, 0, stream>>>(inputs, outputs,
                    exact_queue, count, retry_queue, retry_count, next, words, free_slots, failed,compact_outputs,output_activity);
        } else {
            if (inplace_ratio)
                canonical_point_contacts<false,true><<<workers, 32, 0, stream>>>(inputs, outputs,
                    exact_queue, count, retry_queue, retry_count, next, words, free_slots, failed,compact_outputs,output_activity);
            else
                canonical_point_contacts<false,false><<<workers, 32, 0, stream>>>(inputs, outputs,
                    exact_queue, count, retry_queue, retry_count, next, words, free_slots, failed,compact_outputs,output_activity);
        }
        return;
    }
    if (parallel_geometry_conversions) {
        if(inplace_ratio)
            cooperative_contacts<true,true><<<workers, 32, 0, stream>>>(inputs, outputs,
                exact_queue, count, retry_queue, retry_count, next, words, free_slots, failed,compact_outputs,output_activity);
        else
            cooperative_contacts<true,false><<<workers, 32, 0, stream>>>(inputs, outputs,
                exact_queue, count, retry_queue, retry_count, next, words, free_slots, failed,compact_outputs,output_activity);
    } else {
        if(inplace_ratio)
            cooperative_contacts<false,true><<<workers, 32, 0, stream>>>(inputs, outputs,
                exact_queue, count, retry_queue, retry_count, next, words, free_slots, failed,compact_outputs,output_activity);
        else
            cooperative_contacts<false,false><<<workers, 32, 0, stream>>>(inputs, outputs,
                exact_queue, count, retry_queue, retry_count, next, words, free_slots, failed,compact_outputs,output_activity);
    }
}
void launch_cooperative_exact_contacts(
    const ContactInput* inputs, ContactOutput* outputs,
    const int* exact_queue, const unsigned* count,
    int* retry_queue, unsigned* retry_count, unsigned* next,
    std::uint32_t* words, int* free_slots, unsigned workers,
    int* failed, cudaStream_t stream, CompactDeformableContact* compact_outputs,
    bool parallel_geometry_conversions) {
    launch_cooperative_exact_contacts(inputs,outputs,exact_queue,count,
        retry_queue,retry_count,next,words,free_slots,workers,failed,stream,
        compact_outputs,parallel_geometry_conversions,nullptr);
}
} // namespace ipc_gpu::solver
