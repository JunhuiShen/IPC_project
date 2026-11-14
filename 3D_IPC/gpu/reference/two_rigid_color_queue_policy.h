#pragma once
#include <cstddef>

namespace ipc_gpu::solver::two_rigid_color_queue {
// This predicate supplements GpuSolver's complete scene validation. It does
// not validate arbitrary raw pipeline geometry or grant a public hint API.
template<class Scene> bool eligible_validated_scene(const Scene& h) {
    const auto nodes=h.positions.size();
    if(h.com.size()!=2||h.block_count!=2||h.body_blocks.size()!=2
        ||h.body_blocks[0]<0||h.body_blocks[0]>=2
        ||h.body_blocks[1]<0||h.body_blocks[1]>=2
        ||h.body_blocks[0]==h.body_blocks[1]
        ||h.deformable.size()!=nodes||h.node_body.size()!=nodes
        ||h.node_blocks.size()!=nodes)return false;
    for(std::size_t node=0;node<nodes;++node) {
        const int body=h.node_body[node];
        if(h.deformable[node]||body<0||body>=2
            ||h.node_blocks[node]!=h.body_blocks[static_cast<std::size_t>(body)])return false;
    }
    return true;
}
inline bool omit_color_pairs(bool validated_hint,bool option_enabled,
    bool explicitly_forced,std::size_t pairs,std::size_t packet_capacity,
    int blocks,int colors) {
    return validated_hint&&option_enabled&&!explicitly_forced
        &&blocks==2&&colors==2&&pairs>packet_capacity;
}
} // namespace ipc_gpu::solver::two_rigid_color_queue
