#include "rigid_translation_proxy_owner.h"
#include <array>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
namespace owner=ipc_gpu::solver::rigid_proxy_owner;
using Clipper=ipc_gpu::solver::GpuLinearClipper;
std::size_t proxy_scope_host_consume(const Clipper*,const owner::Storage&,bool);
bool proxy_scope_host_bound(const Clipper*);

// HOST-ONLY authority stand-in, not the production GpuSolver definition/ABI.
// Only this named friend can construct Scope; the consumer is in another TU.
// This standalone program links NO production/CUDA libraries or definitions.
namespace ipc_gpu::solver {
class GpuSolver {
public:
    template<class F>static void scoped(const GpuLinearClipper* clipper,const rigid_proxy_owner::Storage& storage,F&& body) {
        rigid_proxy_owner::Scope scope(clipper,storage);std::forward<F>(body)();
    }
};
}
static_assert(!std::is_constructible<owner::Scope,const Clipper*,const owner::Storage&>::value,
              "untrusted code can construct a positive solver scope");
static_assert(!std::is_copy_constructible<owner::Scope>::value&&!std::is_move_constructible<owner::Scope>::value,
              "scope cannot migrate across owners/threads");
static_assert(!std::is_constructible<owner::Permit,std::size_t>::value,"raw positive permit constructor escaped");
std::size_t checks=0;
struct ExpectedUnwind {};
void require(bool condition,const char* message){++checks;if(!condition)throw std::runtime_error(message);}
owner::Storage storage() {
    owner::Storage s;s.vertices=9;s.proxy_nodes=7;s.bodies=3;s.blocks=9;
    std::array<const void* owner::Storage::*,14> members{{&owner::Storage::positions,&owner::Storage::com,
        &owner::Storage::proposed_com,&owner::Storage::accepted_com,&owner::Storage::accepted,&owner::Storage::boxes,
        &owner::Storage::colors,&owner::Storage::error,&owner::Storage::node_offsets,&owner::Storage::nodes,
        &owner::Storage::node_body,&owner::Storage::body_blocks,&owner::Storage::translate,&owner::Storage::rotate}};
    for(std::size_t i=0;i<members.size();++i)s.*members[i]=reinterpret_cast<const void*>((i+1)*std::uintptr_t(65536));
    return s;
}
int main(){try{
    auto a=storage(),b=a;b.proxy_nodes=6;
    const auto* first=reinterpret_cast<const Clipper*>(std::uintptr_t(16));
    const auto* second=reinterpret_cast<const Clipper*>(std::uintptr_t(32));
    require(!proxy_scope_host_bound(first)&&proxy_scope_host_consume(first,a,true)==0,"raw caller inherited a scope");
    ipc_gpu::solver::GpuSolver::scoped(first,a,[&] {
        require(proxy_scope_host_bound(first),"producer TLS not visible in consumer TU");
        require(!proxy_scope_host_bound(second)&&proxy_scope_host_consume(second,a,true)==0,"foreign clipper borrowed capability");
        require(proxy_scope_host_bound(first),"foreign caller consumed owner capability");
        std::size_t child=99;std::thread thread([&]{child=proxy_scope_host_consume(first,a,true);});thread.join();
        require(child==0&&proxy_scope_host_bound(first),"TLS capability leaked to another thread");
        require(proxy_scope_host_consume(first,a,true)==7,"valid scoped owned storage rejected");
        require(!proxy_scope_host_bound(first)&&proxy_scope_host_consume(first,a,true)==0,"scope was not one-use");
    });
    require(!proxy_scope_host_bound(first),"scope did not restore empty TLS");
    ipc_gpu::solver::GpuSolver::scoped(first,a,[&] {
        require(proxy_scope_host_consume(first,a,false)==0,"disabled flag granted work");
        require(proxy_scope_host_consume(first,a,true)==0,"disabled call left a reusable grant");
    });
    ipc_gpu::solver::GpuSolver::scoped(first,a,[&] {
        require(proxy_scope_host_consume(first,b,true)==0,"stale extent matched capability");
        require(proxy_scope_host_consume(first,a,true)==0,"mismatch did not consume grant");
    });
    auto alias=a;alias.accepted=alias.positions;
    ipc_gpu::solver::GpuSolver::scoped(first,alias,[&] {
        require(proxy_scope_host_consume(first,alias,true)==0,"alias granted parallel writes");
    });
    ipc_gpu::solver::GpuSolver::scoped(first,a,[&] {
        ipc_gpu::solver::GpuSolver::scoped(second,b,[&] {
            require(!proxy_scope_host_bound(first),"outer scope remained active through inner owner");
            require(proxy_scope_host_consume(second,b,true)==6,"nested owner capability rejected");
        });
        require(proxy_scope_host_bound(first)&&proxy_scope_host_consume(first,a,true)==7,"outer scope not restored");
    });
    ipc_gpu::solver::GpuSolver::scoped(first,a,[&] {
        try {
            ipc_gpu::solver::GpuSolver::scoped(first,b,[&] {
                require(proxy_scope_host_consume(first,b,true)==6,"nested same-owner key failed");
                throw ExpectedUnwind{};
            });
        }catch(const ExpectedUnwind&){ }
        require(proxy_scope_host_bound(first)&&proxy_scope_host_consume(first,a,true)==7,
                "exception failed to restore outer capability");
    });
    require(owner::current_scope==nullptr,"scope pointer dangles after unwind");
    // Every pointer field participates in matching; a mismatch consumes scope.
    std::array<const void* owner::Storage::*,14> members{{&owner::Storage::positions,&owner::Storage::com,
        &owner::Storage::proposed_com,&owner::Storage::accepted_com,&owner::Storage::accepted,&owner::Storage::boxes,
        &owner::Storage::colors,&owner::Storage::error,&owner::Storage::node_offsets,&owner::Storage::nodes,
        &owner::Storage::node_body,&owner::Storage::body_blocks,&owner::Storage::translate,&owner::Storage::rotate}};
    for(auto member:members)ipc_gpu::solver::GpuSolver::scoped(first,a,[&] {
        auto wrong=a;wrong.*member=reinterpret_cast<const void*>(reinterpret_cast<std::uintptr_t>(wrong.*member)+4096);
        require(proxy_scope_host_consume(first,wrong,true)==0,"foreign view matched scoped capability");
        require(proxy_scope_host_consume(first,a,true)==0,"foreign-view call left grant reusable");
    });
    std::cout<<"Rigid proxy two-TU scope host PASS: checks="<<checks<<"; owner/TLS/thread/nesting/throw/one-use/alias\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
