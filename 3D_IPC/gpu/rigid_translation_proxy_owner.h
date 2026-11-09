#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ipc_gpu::solver {
class GpuLinearClipper;
class GpuSolver;
namespace rigid_proxy_owner {

// Host metadata only. Validation of CSR contents/unique ownership belongs to
// GpuSolver::validate; only that owner may bind these private storage views.
struct Storage {
    std::size_t vertices=0,proxy_nodes=0,bodies=0,blocks=0;
    const void* positions=nullptr;
    const void* com=nullptr;
    const void* proposed_com=nullptr;
    const void* accepted_com=nullptr;
    const void* accepted=nullptr;
    const void* boxes=nullptr;
    const void* colors=nullptr;
    const void* error=nullptr;
    const void* node_offsets=nullptr;
    const void* nodes=nullptr;
    const void* node_body=nullptr;
    const void* body_blocks=nullptr;
    const void* translate=nullptr;
    const void* rotate=nullptr;
};
struct Range {std::uintptr_t begin=0,end=0;};
inline bool range(const void* pointer,std::size_t count,std::size_t width,
                  std::size_t alignment,Range& output) {
    output={};
    if(!alignment||(alignment&(alignment-1)))return false;
    if(!count)return true;
    if(!pointer||!width||count>std::numeric_limits<std::size_t>::max()/width)return false;
    const auto begin=reinterpret_cast<std::uintptr_t>(pointer);
    const auto bytes=count*width;
    if(begin%alignment||bytes>std::numeric_limits<std::uintptr_t>::max()-begin)return false;
    output={begin,begin+bytes};return true;
}
inline bool disjoint(const Range& a,const Range& b) {
    return a.begin==a.end||b.begin==b.end||a.end<=b.begin||b.end<=a.begin;
}
inline bool storage_is_disjoint(const Storage& s) {
    constexpr auto maximum=static_cast<std::size_t>(std::numeric_limits<int>::max());
    if(!s.vertices||!s.proxy_nodes||!s.bodies||!s.blocks||s.vertices>maximum
        ||s.bodies>maximum||s.blocks>maximum||s.proxy_nodes>s.vertices)return false;
    Range spans[14];
    if(!range(s.positions,s.vertices,3*sizeof(double),alignof(double),spans[0])
        ||!range(s.com,s.bodies,3*sizeof(double),alignof(double),spans[1])
        ||!range(s.proposed_com,s.bodies,3*sizeof(double),alignof(double),spans[2])
        ||!range(s.accepted_com,s.bodies,3*sizeof(double),alignof(double),spans[3])
        ||!range(s.accepted,s.vertices,3*sizeof(double),alignof(double),spans[4])
        ||!range(s.boxes,s.vertices,6*sizeof(double),alignof(double),spans[5])
        ||!range(s.colors,s.blocks,sizeof(int),alignof(int),spans[6])
        ||!range(s.error,1,sizeof(int),alignof(int),spans[7])
        ||!range(s.node_offsets,s.bodies+1,sizeof(std::size_t),alignof(std::size_t),spans[8])
        ||!range(s.nodes,s.proxy_nodes,sizeof(int),alignof(int),spans[9])
        ||!range(s.node_body,s.vertices,sizeof(int),alignof(int),spans[10])
        ||!range(s.body_blocks,s.bodies,sizeof(int),alignof(int),spans[11])
        ||!range(s.translate,s.bodies,sizeof(unsigned char),alignof(unsigned char),spans[12])
        ||!range(s.rotate,s.bodies,sizeof(unsigned char),alignof(unsigned char),spans[13]))return false;
    for(std::size_t i=0;i<14;++i)for(std::size_t j=0;j<i;++j)
        if(!disjoint(spans[i],spans[j]))return false;
    return true;
}
inline bool same(const Storage& a,const Storage& b) {
    return a.vertices==b.vertices&&a.proxy_nodes==b.proxy_nodes&&a.bodies==b.bodies&&a.blocks==b.blocks
        &&a.positions==b.positions&&a.com==b.com&&a.proposed_com==b.proposed_com
        &&a.accepted_com==b.accepted_com&&a.accepted==b.accepted&&a.boxes==b.boxes
        &&a.colors==b.colors&&a.error==b.error&&a.node_offsets==b.node_offsets&&a.nodes==b.nodes
        &&a.node_body==b.node_body&&a.body_blocks==b.body_blocks&&a.translate==b.translate&&a.rotate==b.rotate;
}

// Only a solver-owned, one-use Scope can create a nonempty permit. Raw/internal
// callers can construct only the empty legacy permit. Neither this token nor
// its Scope becomes a device kernel argument.
class Scope;
extern thread_local Scope* current_scope;
class Permit {
    friend class Scope;
    std::size_t proxy_nodes_=0;
    explicit Permit(std::size_t nodes):proxy_nodes_(nodes){}
public:
    Permit()=default;
    std::size_t proxy_nodes()const{return proxy_nodes_;}
};
// Stack lifetime is host-only; one external TLS definition is in the caller
// CU. No changed definition of the existing public GpuLinearClipper class.
class Scope {
    friend class ::ipc_gpu::solver::GpuSolver;
    const GpuLinearClipper* owner_;
    Storage storage_;
    Scope* previous_;
    bool available_=true;
    Scope(const GpuLinearClipper* owner,const Storage& storage) noexcept
        :owner_(owner),storage_(storage),previous_(current_scope) {
        // POD construction cannot throw; publish TLS only after it completes.
        current_scope=this;
    }
public:
    ~Scope() noexcept {current_scope=previous_;}
    Scope(const Scope&)=delete;
    Scope& operator=(const Scope&)=delete;
    static const Storage* binding(const GpuLinearClipper* owner) noexcept {
        return current_scope&&current_scope->owner_==owner&&current_scope->available_
            ?&current_scope->storage_:nullptr;
    }
    static Permit consume(const GpuLinearClipper* owner,const Storage& current,bool enabled) noexcept {
        auto* scope=current_scope;
        if(!scope||scope->owner_!=owner)return {};
        const bool available=scope->available_;
        scope->available_=false;
        if(!available||!enabled||!same(current,scope->storage_)||!storage_is_disjoint(current))return {};
        return Permit(current.proxy_nodes);
    }
};
} // namespace rigid_proxy_owner
} // namespace ipc_gpu::solver
