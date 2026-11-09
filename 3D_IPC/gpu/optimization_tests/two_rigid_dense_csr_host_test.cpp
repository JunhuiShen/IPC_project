#include "two_rigid_dense_csr_policy.h"
#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace dense=ipc_gpu::solver::two_rigid_dense_csr;
namespace {
struct Contact {int vertices[4];int kind;};
struct Case {std::string name;std::vector<Contact> contacts;std::vector<int> blocks;unsigned reason;};
void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
std::vector<Case> dense_cases() {
    const std::vector<int> blocks{0,0,1,1,1,0};
    const Contact pt{{0,2,3,4},0},ee{{0,1,2,3},1};
    std::vector<Case> out;
    for(std::size_t count:{0u,1u,2u,3u,4u,127u,128u,129u,257u})
        for(int mode=0;mode<3;++mode) {
            Case c{"ordered-"+std::to_string(count)+"-"+std::to_string(mode),{},blocks,0};
            for(std::size_t i=0;i<count;++i)c.contacts.push_back(mode==0||(mode==2&&i<count/2)?pt:ee);
            out.push_back(c);
        }
    out.push_back({"descending-kind",{ee,pt},blocks,dense::NonMonotoneKind});
    auto bad=ee;bad.kind=2;out.push_back({"invalid-kind",{bad},blocks,dense::InvalidKind});
    out.push_back({"invalid-then-descending",{bad,pt},blocks,dense::InvalidKind|dense::NonMonotoneKind});
    bad=ee;bad.kind=-1;out.push_back({"negative-kind",{bad},blocks,dense::InvalidKind});
    bad.kind=-2;out.push_back({"negative-descending-kind",{bad},blocks,dense::InvalidKind|dense::NonMonotoneKind});
    bad=ee;bad.vertices[0]=-1;out.push_back({"negative-vertex",{bad},blocks,dense::InvalidVertex});
    bad=ee;bad.vertices[0]=static_cast<int>(blocks.size());out.push_back({"past-end-vertex",{bad},blocks,dense::InvalidVertex});
    bad=ee;bad.vertices[0]=INT_MAX;out.push_back({"huge-vertex",{bad},blocks,dense::InvalidVertex});
    auto bad_blocks=blocks;bad_blocks[0]=2;out.push_back({"third-block",{ee},bad_blocks,dense::InvalidBlock});
    bad_blocks=blocks;bad_blocks[0]=-1;out.push_back({"negative-block",{ee},bad_blocks,dense::InvalidBlock});
    bad=Contact{{0,1,0,1},0};out.push_back({"same-block",{bad},blocks,dense::MissingBlock});
    bad_blocks=blocks;bad_blocks[0]=bad_blocks[1]=-1;
    out.push_back({"no-valid-first-block",{ee},bad_blocks,dense::InvalidBlock|dense::MissingBlock});
    out.push_back({"missing-vertex-array",{ee},{},dense::InvalidVertex|dense::MissingBlock});
    // All14 cross-owner role patterns, including a stronger superset of the
    // homogeneous NT/EE geometry admitted by the real validated broad phase.
    Case patterns{"all-cross-role-patterns",{},blocks,0};
    for(int kind=0;kind<2;++kind)for(unsigned mask=1;mask<15;++mask) {
        Contact c{};c.kind=kind;
        for(unsigned role=0;role<4;++role)c.vertices[role]=((mask>>role)&1u)?2:0;
        patterns.contacts.push_back(c);
    }
    out.push_back(patterns);
    auto swapped=patterns;swapped.name="swapped-block-IDs";
    for(auto& block:swapped.blocks)block=1-block;
    out.push_back(swapped);
    return out;
}
unsigned inspect_case(const Case& c) {
    unsigned reason=0;
    for(std::size_t i=0;i<c.contacts.size();++i)
        reason|=dense::inspect(c.contacts[i],i?c.contacts[i-1].kind:-1,c.blocks.data(),c.blocks.size());
    return reason;
}
// Independent consumed-row oracle: select each pair once for each block it
// contains, then sort by(kind,pair). Do not compare unused sentinel tails.
std::array<std::vector<int>,2> generic_rows(const Case& c) {
    std::array<std::vector<int>,2> rows;
    for(int block=0;block<2;++block) {
        for(std::size_t pair=0;pair<c.contacts.size();++pair) {
            bool selected=false;
            for(int vertex:c.contacts[pair].vertices) {
                require(vertex>=0&&static_cast<std::size_t>(vertex)<c.blocks.size(),"oracle vertex unsupported");
                selected=selected||c.blocks[static_cast<std::size_t>(vertex)]==block;
            }
            if(selected)rows[block].push_back(static_cast<int>(pair));
        }
        std::sort(rows[block].begin(),rows[block].end(),[&](int a,int b){
            return c.contacts[a].kind!=c.contacts[b].kind?c.contacts[a].kind<c.contacts[b].kind:a<b;
        });
    }
    return rows;
}
void host_inspection() {
    for(const auto& c:dense_cases()) {
        const auto before_contacts=c.contacts;const auto before_blocks=c.blocks;
        require(inspect_case(c)==c.reason,c.name+" reason mismatch");
        require(c.blocks==before_blocks&&c.contacts.size()==before_contacts.size()
            &&(c.contacts.empty()||std::memcmp(c.contacts.data(),before_contacts.data(),c.contacts.size()*sizeof(Contact))==0),c.name+" input mutated");
        if(c.reason==0) {
            const auto rows=generic_rows(c);
            for(const auto& row:rows) {
                require(row.size()==c.contacts.size(),c.name+" cross-block row omitted a pair");
                for(std::size_t i=0;i<row.size();++i)
                    require(row[i]==static_cast<int>(i),c.name+" kind/pair order is not natural");
            }
        }
    }
    // A valid but nonmonotone sequence must preserve the generic reordered
    // rows, not silently accept the natural dense ordering.
    const auto c=dense_cases()[27];const auto rows=generic_rows(c);
    require(c.name=="descending-kind"&&rows[0]==std::vector<int>({1,0})
        &&rows[1]==rows[0],"generic fallback ordering premise is vacuous");
}
void host_gate_and_capacity() {
    struct Counts {std::size_t n,cap;bool valid;};
    const auto imax=static_cast<std::size_t>(INT_MAX);
    const std::array<Counts,7> counts{{{0,3,false},{3,3,false},{4,3,true},
        {65537,65536,true},{imax,1,true},{imax+1,1,false},
        {std::numeric_limits<std::size_t>::max(),1,false}}};
    for(unsigned mask=0;mask<8;++mask)for(int blocks:{-1,0,1,2,3})for(const auto& c:counts) {
        const bool expected=mask==3&&blocks==2&&c.valid;
        require(dense::eligible(mask&1u,mask&2u,mask&4u,c.n,c.cap,blocks)==expected,"eligibility truth table mismatch");
    }
    struct Epoch {bool dense;std::size_t required,expected_capacity;bool release;};
    // Generic4N32 -> dense16; dense highwater stays16 at14/12; grow18;
    // generic16 retains18; enteringdense8 must release that generic highwater.
    const std::array<Epoch,8> epochs{{{false,32,32,false},{true,16,16,true},
        {true,14,16,false},{true,12,16,false},{true,18,18,false},
        {false,16,18,false},{true,8,8,true},{true,8,8,false}}};
    std::size_t retained=0;bool previous=false;
    for(const auto& e:epochs) {
        const bool release=e.dense&&dense::release_generic_capacity(previous,retained,e.required);
        require(release==e.release,"mode transition release decision mismatch");
        if(release)retained=0;
        retained=std::max(retained,e.required);previous=e.dense;
        require(retained==e.expected_capacity,"dense highwater changed on decrease");
    }
}
void host_kind_packets() {
    std::size_t protocols=0;
    for(std::size_t count:{0u,1u,2u,3u,127u,128u,129u,257u,513u})
        for(std::size_t capacity:{1u,2u,3u,127u,128u,129u,256u})
            for(std::size_t boundary=0;boundary<=count;++boundary)for(int kind=0;kind<2;++kind) {
                const auto span=dense::kind_packet_span(true,count,capacity,boundary,kind);
                require(span.first<=span.last&&span.last<=count,"kind span outside contact domain");
                std::vector<std::size_t> expected_pairs,actual_pairs;
                std::vector<std::pair<std::size_t,std::size_t>> expected_packets,actual_packets;
                // Independent legacy schedule: enumerate every original packet
                // and retain it iff at least one record has the requested kind.
                for(std::size_t first=0;first<count;) {
                    const auto last=first+std::min(capacity,count-first);
                    bool any=false;
                    for(auto pair=first;pair<last;++pair) {
                        const int actual_kind=pair<boundary?0:1;
                        if(actual_kind==kind){any=true;expected_pairs.push_back(pair);}
                    }
                    if(any)expected_packets.emplace_back(first,last);
                    first=last;
                }
                for(auto first=span.first;first<span.last;) {
                    const auto last=first+std::min(capacity,span.last-first);
                    actual_packets.emplace_back(first,last);
                    for(auto pair=first;pair<last;++pair)
                        if((pair<boundary?0:1)==kind)actual_pairs.push_back(pair);
                    first=last;
                }
                require(actual_packets==expected_packets,"executed packet arguments differ from the legacy nonempty subset");
                require(actual_pairs==expected_pairs,"kind schedule changed pair coverage or accumulation order");
                const auto generic=dense::kind_packet_span(false,count,capacity,boundary,kind);
                require(generic.first==0&&generic.last==count,"uncertified input omitted a packet");
                ++protocols;
            }
    const auto maximum=std::numeric_limits<std::size_t>::max();
    for(std::size_t count:{static_cast<std::size_t>(INT_MAX),maximum})
        for(std::size_t capacity:{std::size_t(1),std::size_t(128),maximum})
            for(std::size_t boundary:{std::size_t(0),std::size_t(1),count-1,count})
                for(int kind=0;kind<2;++kind) {
                    const auto span=dense::kind_packet_span(true,count,capacity,boundary,kind);
                    require(span.first<=span.last&&span.last<=count,"wide kind range overflow");
                    if(span.first<span.last) {
                        require(span.first%capacity==0,"wide kind start is not a legacy packet boundary");
                        require(span.last==count||span.last%capacity==0,"wide kind end is not a legacy packet boundary");
                    }
                }
    for(int kind:{-1,2}) {
        const auto span=dense::kind_packet_span(true,257,128,129,kind);
        require(span.first==0&&span.last==257,"unsupported kind lost the complete fallback");
    }
    for(const auto& span:{dense::kind_packet_span(true,257,0,129,0),
        dense::kind_packet_span(true,257,128,258,1)})
        require(span.first==0&&span.last==257,"invalid metadata lost the complete fallback");
    std::cout<<"Kind packet protocols "<<protocols<<" preserve original packet arguments and ordered contact coverage\n";
}
}
int main() {
    try{host_inspection();host_gate_and_capacity();host_kind_packets();std::cout<<"Two-rigid dense CSR host PASS: "
        <<dense_cases().size()<<" inspected layouts, consumed kind/pair rows, gate boundaries and8 capacity epochs\n";return 0;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
