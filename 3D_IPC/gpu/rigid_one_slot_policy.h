#pragma once
#include <cstddef>
#ifdef __CUDACC__
#define IPC_ONE_SLOT_HD __host__ __device__
#else
#define IPC_ONE_SLOT_HD
#endif
namespace ipc_gpu::solver::rigid_one_slot {
IPC_ONE_SLOT_HD constexpr bool proper_colors(const int* colors,const int* offsets,
    const int* blocks,int block_count,int color_count) {
    return colors&&offsets&&blocks&&block_count==2&&color_count==2
        &&colors[0]>=0&&colors[0]<2&&colors[1]>=0&&colors[1]<2
        &&colors[0]!=colors[1]&&offsets[0]==0&&offsets[1]==1&&offsets[2]==2
        &&blocks[0]>=0&&blocks[0]<2&&blocks[1]>=0&&blocks[1]<2
        &&blocks[0]!=blocks[1]&&colors[blocks[0]]==0&&colors[blocks[1]]==1;
}
inline constexpr bool eligible(bool trusted,bool enabled,bool dense,bool pruning,
    bool graph_proven,bool original,bool frozen,bool tiled,bool cached,
    bool rigid_phase,std::size_t bodies,std::size_t nodes,std::size_t validated_nodes,
    bool matching_node_blocks) {
    return trusted&&enabled&&dense&&!pruning&&graph_proven&&original&&!frozen
        &&tiled&&cached&&rigid_phase&&bodies==2&&nodes==validated_nodes&&matching_node_blocks;
}
IPC_ONE_SLOT_HD constexpr std::size_t term_index(bool one_slot,std::size_t slot,
                                                 unsigned first_owned_role) {
    return one_slot?slot:4*slot+first_owned_role;
}
inline constexpr bool release_on_transition(bool old_one_slot,bool one_slot,
    std::size_t retained,std::size_t required) {
    return old_one_slot!=one_slot&&retained>required;
}
} // namespace ipc_gpu::solver::rigid_one_slot
#undef IPC_ONE_SLOT_HD
