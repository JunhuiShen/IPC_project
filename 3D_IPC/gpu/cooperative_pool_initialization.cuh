#pragma once
// Private initialization experiment: no arithmetic, gate, capacity or reset change.
#include "contact_cooperative.cuh"
namespace ipc_gpu::solver::pool_initialization_study {
namespace ex=exact;
namespace cp=contact_cooperative_detail;
static_assert(cp::capacity==256&&cp::owner_slots==512&&cp::helper_slots==128
              &&cp::active_lanes==9&&cp::slots_per_worker==1536);
// Exactly the original32-thread one-dimensional CTA. The caller's original B0
// must occur before ANY Integer construction, admission vote or pool lookup.
// Caller launch bounds/allocation ownership remain the original ones.
__device__ inline void initialize(ex::DevicePool& pool,std::uint32_t* base_words,
                                 int* base_slots,unsigned worker) {
    const unsigned lane=threadIdx.x;
    const std::size_t base=std::size_t(worker)*cp::slots_per_worker;
    if(lane<cp::active_lanes) {
        const int private_slots=lane==0?cp::owner_slots:cp::helper_slots;
        const std::size_t offset=base+(lane==0?0:cp::owner_slots+(lane-1)*cp::helper_slots);
        // The same metadata as original initialize_pool, including failed state
        // and null installation when either supplied storage base is null.
        ex::thread_pool_slot()=nullptr;
        pool.words=base_words?base_words+offset*cp::capacity:nullptr;
        pool.free_slots=base_slots?base_slots+offset:nullptr;
        pool.words_per_slot=cp::capacity;pool.slot_count=private_slots;
        pool.available=0;pool.peak=0;
        pool.failed=!pool.words||!pool.free_slots||pool.words_per_slot<=0||pool.slot_count<=0;
        if(!pool.failed)pool.available=private_slots;
    }
    if(base_words&&base_slots) {
        for(unsigned index=lane;index<cp::slots_per_worker;index+=32) {
            // Original owner identities0..511; each helper's own0..127.
            base_slots[base+index]=index<cp::owner_slots?int(index):int((index-cp::owner_slots)%cp::helper_slots);
        }
    }
    if(lane<cp::active_lanes&&!pool.failed)ex::thread_pool_slot()=&pool;
    // No new barrier: original B0 publishes ALL identities and registrations.
}
} // namespace ipc_gpu::solver::pool_initialization_study
