#pragma once
#include "broadphase.h"
#include "contact_worklists.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ipc_gpu::solver {

// Host launch metadata only. No device read, arithmetic, allocation of device
// storage, changed pair membership, or implicit last-assembled-color state.
// A pipeline rebuild invalidates this binding before doing any fallible work.
// Publication copies counts from that rebuild's already-downloaded offsets.
class ContactColorLaunchBinding {
    bool ready_=false, exact_=false;
    std::uint64_t generation_=0,published_generation_=0;
    BroadPhaseView broad_{};
    ColorContactWorklist work_{};
    std::vector<std::size_t> counts_;

    static bool same_view(const BroadPhaseView& a,const BroadPhaseView& b) {
        return a.contacts==b.contacts&&a.contact_count==b.contact_count
            &&a.block_colors==b.block_colors&&a.color_count==b.color_count
            &&a.color_blocks==b.color_blocks&&a.color_offsets==b.color_offsets;
    }
    static bool same_worklist(const ColorContactWorklist& a,const ColorContactWorklist& b) {
        return a.offsets==b.offsets&&a.pair_indices==b.pair_indices
            &&a.max_color_pairs==b.max_color_pairs&&a.box_separated==b.box_separated;
    }
public:
    void begin_rebuild() {
        ready_=false;exact_=false;counts_.clear();
        if(generation_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("contact color launch generation exhausted");
        ++generation_;
    }
    void publish(const BroadPhaseView& broad,ColorContactWorklist work,
                 const std::vector<std::size_t>& host_offsets,bool enabled) {
        if(!generation_||ready_)throw std::logic_error("contact color launch publication requires a new rebuild");
        if(broad.color_count<0)throw std::invalid_argument("negative contact color count");
        const bool exact=enabled&&work.offsets;
        if(exact) {
            const auto colors=static_cast<std::size_t>(broad.color_count);
            if(host_offsets.size()!=colors+1||host_offsets.front()!=0
                ||!std::is_sorted(host_offsets.begin(),host_offsets.end())
                ||(host_offsets.back()&&!work.pair_indices)
                ||broad.contact_count>std::numeric_limits<std::size_t>::max()/4
                ||host_offsets.back()>4*broad.contact_count)
                throw std::invalid_argument("contact color launch offsets do not match this rebuild");
            std::size_t maximum=0;
            for(std::size_t color=0;color<colors;++color) {
                const auto count=host_offsets[color+1]-host_offsets[color];
                // emit_color_pairs emits each candidate once for a color.
                if(count>broad.contact_count)
                    throw std::invalid_argument("contact color launch count exceeds candidates");
                maximum=std::max(maximum,count);
            }
            if(maximum!=work.max_color_pairs)
                throw std::invalid_argument("contact color launch maximum does not match offsets");
            counts_.resize(colors);
            for(std::size_t color=0;color<colors;++color)
                counts_[color]=host_offsets[color+1]-host_offsets[color];
        }
        broad_=broad;work_=work;exact_=exact;
        published_generation_=generation_;ready_=true;
    }
    ColorContactWorklist for_color(const BroadPhaseView& broad,
                                   ColorContactWorklist work,int color) const {
        if(!ready_||published_generation_!=generation_)
            throw std::logic_error("contact color launch requires a completed rebuild");
        if(color<0||color>=broad_.color_count)
            throw std::out_of_range("contact color launch index is invalid");
        if(!same_view(broad,broad_)||!same_worklist(work,work_))
            throw std::logic_error("contact color launch metadata belongs to another rebuild");
        if(exact_)work.max_color_pairs=counts_[static_cast<std::size_t>(color)];
        return work;
    }
    std::uint64_t generation()const{return generation_;}
};
} // namespace ipc_gpu::solver
