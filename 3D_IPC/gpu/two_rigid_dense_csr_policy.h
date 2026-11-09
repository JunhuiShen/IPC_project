#pragma once
#include <cstddef>
#include <limits>

#if defined(__CUDACC__)
#define IPC_TWO_RIGID_DENSE_HD __host__ __device__
#else
#define IPC_TWO_RIGID_DENSE_HD
#endif
namespace ipc_gpu::solver::two_rigid_dense_csr {
enum Unsupported : unsigned {
    InvalidKind=1u, NonMonotoneKind=2u, InvalidVertex=4u,
    InvalidBlock=8u, MissingBlock=16u
};
// The caller additionally owns the validated two-rigid scene hint. A pruning
// cache is deliberately unsupported, even if its particular flags are all0.
inline bool eligible(bool trusted,bool enabled,bool pruning,std::size_t pairs,
    std::size_t capacity,int blocks) {
    return trusted&&enabled&&!pruning&&blocks==2&&pairs>capacity
        &&pairs<=static_cast<std::size_t>(std::numeric_limits<int>::max())
        &&pairs<=std::numeric_limits<std::size_t>::max()/2;
}
template<class Contact>
IPC_TWO_RIGID_DENSE_HD inline unsigned inspect(const Contact& contact,
    int previous_kind,const int* vertex_blocks,std::size_t nodes) {
    unsigned unsupported=0,mask=0;
    if(contact.kind!=0&&contact.kind!=1)unsupported|=InvalidKind;
    if(previous_kind>contact.kind)unsupported|=NonMonotoneKind;
    for(unsigned role=0;role<4;++role) {
        const int vertex=contact.vertices[role];
        if(vertex<0||static_cast<std::size_t>(vertex)>=nodes) {
            unsupported|=InvalidVertex;continue;
        }
        const int block=vertex_blocks[vertex];
        if(block<0||block>1)unsupported|=InvalidBlock;
        else mask|=1u<<static_cast<unsigned>(block);
    }
    if(mask!=3u)unsupported|=MissingBlock;
    return unsupported;
}
// Shrink once when entering dense mode from a generic epoch. Within dense
// mode retain the high-water capacity across decreasing N; reserve grows it.
inline bool release_generic_capacity(bool previously_dense,std::size_t retained,
    std::size_t required) {
    return !previously_dense&&retained>required;
}

struct PacketSpan {std::size_t first,last;};
// Omit only complete legacy packets containing no contact of this kind.
// The boundary packet is deliberately retained for both kinds: every kernel
// that still runs receives exactly its old global first/last pair indices.
// The split is admissible only after the dense preflight certifies all kinds
// and their nondecreasing order. Unsupported/raw/pruned epochs use [0,N).
inline PacketSpan kind_packet_span(bool certified,std::size_t count,
    std::size_t capacity,std::size_t point_triangle_end,int kind) {
    if(!certified||!capacity||point_triangle_end>count||(kind!=0&&kind!=1))
        return {0,count};
    if(kind==0) {
        if(!point_triangle_end)return {0,0};
        const auto remainder=point_triangle_end%capacity;
        const auto padding=remainder?capacity-remainder:0;
        const auto available=count-point_triangle_end;
        return {0,point_triangle_end+(padding<available?padding:available)};
    }
    if(point_triangle_end==count)return {count,count};
    return {(point_triangle_end/capacity)*capacity,count};
}
} // namespace ipc_gpu::solver::two_rigid_dense_csr
#undef IPC_TWO_RIGID_DENSE_HD
