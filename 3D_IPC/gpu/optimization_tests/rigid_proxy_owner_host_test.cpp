#include "rigid_translation_proxy_owner.h"
#include <array>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace owner=ipc_gpu::solver::rigid_proxy_owner;
static_assert(std::is_default_constructible<owner::Permit>::value,"raw callers retain empty permit");
static_assert(!std::is_constructible<owner::Permit,std::size_t>::value,
              "untrusted callers must not construct a nonempty owner permit");
std::size_t checks=0;
void require(bool condition,const char* message){++checks;if(!condition)throw std::runtime_error(message);}
std::array<const void* owner::Storage::*,14> members{{
    &owner::Storage::positions,&owner::Storage::com,&owner::Storage::proposed_com,
    &owner::Storage::accepted_com,&owner::Storage::accepted,&owner::Storage::boxes,
    &owner::Storage::colors,&owner::Storage::error,&owner::Storage::node_offsets,
    &owner::Storage::nodes,&owner::Storage::node_body,&owner::Storage::body_blocks,
    &owner::Storage::translate,&owner::Storage::rotate}};
owner::Storage storage() {
    owner::Storage s;s.vertices=9;s.proxy_nodes=7;s.bodies=3;s.blocks=9;
    const std::uintptr_t stride=sizeof(std::uintptr_t)>=8?static_cast<std::uintptr_t>(std::uint64_t(1)<<40):65536;
    for(std::size_t i=0;i<members.size();++i)s.*members[i]=reinterpret_cast<const void*>((i+1)*stride);
    return s;
}
int main(){try{
    auto good=storage();require(owner::storage_is_disjoint(good),"valid disjoint storage rejected");
    require(owner::same(good,good),"identical current binding does not match");
    require(owner::Permit{}.proxy_nodes()==0,"default raw permit grants work");
    for(auto member:members) {
        auto bad=good;bad.*member=nullptr;
        require(!owner::storage_is_disjoint(bad),"nonnull required span accepted null");
        bad=good;bad.*member=reinterpret_cast<const void*>(reinterpret_cast<std::uintptr_t>(bad.*member)+4096);
        require(!owner::same(bad,good),"changed view matched stale binding");
    }
    for(std::size_t a=0;a<members.size();++a)for(std::size_t b=0;b<a;++b) {
        auto bad=good;bad.*members[a]=bad.*members[b];
        require(!owner::storage_is_disjoint(bad),"aliased owner arrays accepted");
    }
    auto bad=good;bad.accepted=reinterpret_cast<const void*>(reinterpret_cast<std::uintptr_t>(good.positions)+8);
    require(!owner::storage_is_disjoint(bad),"partial position/endpoint alias accepted");
    bad=good;bad.accepted_com=reinterpret_cast<const void*>(reinterpret_cast<std::uintptr_t>(good.com)+8);
    require(!owner::storage_is_disjoint(bad),"partial COM/output alias accepted");
    bad=good;bad.boxes=reinterpret_cast<const void*>(reinterpret_cast<std::uintptr_t>(good.accepted)+8);
    require(!owner::storage_is_disjoint(bad),"box/output alias accepted");
    bad=good;bad.nodes=reinterpret_cast<const void*>(reinterpret_cast<std::uintptr_t>(good.accepted)+4);
    require(!owner::storage_is_disjoint(bad),"CSR/output alias accepted");
    for(auto count:{&owner::Storage::vertices,&owner::Storage::proxy_nodes,&owner::Storage::bodies,&owner::Storage::blocks}) {
        bad=good;bad.*count=0;
        require(!owner::storage_is_disjoint(bad),"empty owner capability admitted");
        bad=good;bad.*count+=1;
        require(!owner::same(bad,good),"changed extent matched stale binding");
    }
    bad=good;bad.proxy_nodes=10;
    require(!owner::storage_is_disjoint(bad),"proxy count exceeds validated unique nodes");
    const auto int_max=static_cast<std::size_t>(std::numeric_limits<int>::max());
    for(auto count:{&owner::Storage::vertices,&owner::Storage::bodies,&owner::Storage::blocks}) {
        bad=good;bad.*count=int_max+1;
        require(!owner::storage_is_disjoint(bad),"unsupported public integer extent admitted");
    }
    if(sizeof(std::uintptr_t)>=8) {
        auto huge=good;huge.vertices=huge.proxy_nodes=huge.bodies=huge.blocks=int_max;
        require(owner::storage_is_disjoint(huge),"valid INT_MAX range arithmetic rejected");
    }
    owner::Range a,b;
    require(owner::range(reinterpret_cast<const void*>(4096),2,8,8,a),"aligned span rejected");
    require(owner::range(reinterpret_cast<const void*>(4112),1,8,8,b)&&owner::disjoint(a,b),"adjacent spans overlap");
    require(owner::range(reinterpret_cast<const void*>(4104),1,8,8,b)&&!owner::disjoint(a,b),"partial overlap missed");
    require(!owner::range(reinterpret_cast<const void*>(4097),1,8,8,a),"misalignment admitted");
    require(!owner::range(reinterpret_cast<const void*>(4096),1,8,0,a),"zero alignment admitted");
    require(!owner::range(reinterpret_cast<const void*>(4096),1,8,3,a),"nonpower alignment admitted");
    require(!owner::range(reinterpret_cast<const void*>(4096),std::numeric_limits<std::size_t>::max(),8,8,a),
            "byte multiplication overflow admitted");
    require(!owner::range(reinterpret_cast<const void*>(std::numeric_limits<std::uintptr_t>::max()-7),2,8,8,a),
            "pointer endpoint overflow admitted");
    require(owner::range(nullptr,0,8,8,a)&&owner::disjoint(a,b),"empty interval not harmless");
    std::cout<<"Rigid proxy private-owner host acceptance PASS: checks="<<checks
             <<"; no CUDA or numerical kernel simulation\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
