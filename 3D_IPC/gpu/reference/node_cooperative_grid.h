#pragma once
#include <cstddef>
#include <limits>

namespace ipc_gpu::solver::node_cooperative_grid {
constexpr unsigned block_width=128;
struct Decision {
    unsigned grid=0;
    std::size_t required=0;
    bool valid=false;
};
// The incoming grid is the existing actual-kernel residency ceiling. Shrink
// only after preserving all per-node threads and block=worker exact arenas.
constexpr Decision choose(unsigned incoming,std::size_t nodes,int workers,
    unsigned preferred) noexcept {
    if(!incoming||!nodes||workers<=0)return {incoming,0,false};
    const auto node_blocks=nodes/block_width+(nodes%block_width!=0);
    const auto worker_blocks=static_cast<std::size_t>(workers);
    const auto required=node_blocks>worker_blocks?node_blocks:worker_blocks;
    if(required>incoming)return {incoming,required,false};
    const auto desired=preferred?static_cast<std::size_t>(preferred):static_cast<std::size_t>(incoming);
    const auto covered=desired>required?desired:required;
    const auto selected=covered<incoming?covered:static_cast<std::size_t>(incoming);
    return {static_cast<unsigned>(selected),required,true};
}
} // namespace ipc_gpu::solver::node_cooperative_grid
